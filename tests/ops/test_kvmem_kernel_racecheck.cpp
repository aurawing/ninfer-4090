#include "partial_attention_reference.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/kvmem_score.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::partial_attention_reference;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
const char* name(Format f) {
    return f == Format::Bf16 ? "bf16" : f == Format::Int8 ? "int8" : "rk4";
}
struct Cache {
    EncodedCache host;
    DeviceBuffer k, v, ks, vs;
    Cache(Format f) : host(f, 256), k(to_device(host.k)), v(to_device(host.v)),
        ks(to_device(host.ks)), vs(to_device(host.vs)) {}
    PagedKVLayerView view(int first, int count) {
        const bool plain = host.format == Format::Bf16, packed = host.format == Format::Rk4v4E8;
        const auto bytes = std::size_t(host.code_dim) * 64 * 4 * (plain ? 2 : 1);
        PagedKVLayerView c;
        if (count) {
            c.k_pages = Tensor(static_cast<std::byte*>(k.p) + first * bytes,
                plain ? DType::BF16 : packed ? DType::U8 : DType::I8, {host.code_dim,64,4,count});
            c.v_pages = Tensor(static_cast<std::byte*>(v.p) + first * bytes, c.k_pages.dtype,
                {host.code_dim,64,4,count});
            if (!plain) {
                c.k_scale_pages = Tensor(static_cast<std::uint16_t*>(ks.p) + first * 1024,
                    DType::FP16, {4,64,4,count});
                c.v_scale_pages = Tensor(static_cast<std::uint16_t*>(vs.p) + first * 1024,
                    DType::FP16, {4,64,4,count});
            }
        }
        c.head_dim=256; c.num_kv_heads=4; c.dtype=plain ? DType::BF16 : DType::I8;
        c.quant_group=plain ? 0 : 64;
        c.packed_k=c.packed_v=c.rotate_k=c.rotate_v=c.e8_lattice=packed;
        return c;
    }
};
struct Configuration { Format format; int tokens, splits; };
// Exact T/split tuples observed at successful layer-0 completion in the full
// external owner inventory. Logical/physical page count is reduced, not T/split.
constexpr Configuration configurations[]{
    {Format::Bf16,1,2}, {Format::Bf16,1,12}, {Format::Bf16,1,18}, {Format::Bf16,1,32},
    {Format::Bf16,2,2}, {Format::Bf16,3,12}, {Format::Bf16,4,12},
    {Format::Bf16,16,1}, {Format::Bf16,32,1}, {Format::Bf16,64,1},
    {Format::Int8,1,12}, {Format::Int8,3,13}, {Format::Int8,4,13},
    {Format::Int8,32,1}, {Format::Int8,63,1}, {Format::Int8,64,1},
    {Format::Rk4v4E8,1,12}, {Format::Rk4v4E8,3,13}, {Format::Rk4v4E8,4,13},
    {Format::Rk4v4E8,32,1}, {Format::Rk4v4E8,63,1}, {Format::Rk4v4E8,64,1},
};
int partial_case(Configuration cfg, bool mixed) {
    std::cout << "[kernel-case] partial format=" << name(cfg.format) << " T=" << cfg.tokens
        << " split=" << cfg.splits << " mixed=" << mixed << " valid_keys=77 tail=13\n" << std::flush;
    Cache cache(cfg.format);
    const int resident_count=mixed ? 2 : 4;
    auto resident=cache.view(0,resident_count), staging=cache.view(resident_count,4-resident_count);
    constexpr int frontier=64013;
    std::vector<ops::AttentionPageAccess> pages{{0,1},{1000,3}};
    auto prefix=ops::attention_access_prefix(pages,frontier,resident_count,4-resident_count);
    require(prefix.back()==77,"small fragmented/tail prefix");
    auto dp=to_device(pages), df=to_device(prefix);
    auto query=queries(cfg.tokens);
    auto dq=to_device(query);
    std::vector<std::int32_t> positions(cfg.tokens);
    std::iota(positions.begin(),positions.end(),frontier-cfg.tokens);
    auto dpos=to_device(positions);
    std::vector<int> key_positions(256,std::numeric_limits<int>::max());
    for (auto page:pages) for(int j=0;j<64;++j) {
        const int ordinal=page.logical_page*64+j;
        if(ordinal<frontier) key_positions[page.physical_page*64+j]=ordinal;
    }
    const auto expected=oracle(cache.host,query,positions,{},key_positions);
    Parts partial(cfg.tokens,cfg.splits), state(cfg.tokens,1);
    DeviceBuffer output(query.size()*2);
    Tensor tq(dq.p,DType::BF16,{256,24,cfg.tokens}), tp(dpos.p,DType::I32,{cfg.tokens}),
        ta(dp.p,DType::I32,{2,2}), tf(df.p,DType::I32,{3}), out(output.p,DType::BF16,{256,24,cfg.tokens});
    const auto invoke=cfg.tokens>4 ? ops::gqa_attention_partial_prefill : ops::gqa_attention_partial_decode;
    // Same owner carry/finalize branches, and a second pass verifies that the
    // carried FP32 state is normalized once, rather than averaging BF16 outputs.
    for(int pass=0;pass<2;++pass) {
        invoke(tq,tp,.0625f,resident,staging,ta,tf,frontier,cfg.splits,partial.tensors,nullptr);
        ops::attention_partial_lse_accumulate(partial.tensors,state.tensors,pass==0,nullptr,
            cfg.format==Format::Rk4v4E8 ? nullptr : &out);
    }
    if(cfg.format==Format::Rk4v4E8) ops::attention_partial_finalize_rotated(state.tensors,out,nullptr);
    cuda_synchronize();
    const auto actual_bits=from_device<std::uint16_t>(output,query.size());
    std::vector<double> actual(actual_bits.size());
    std::transform(actual_bits.begin(),actual_bits.end(),actual.begin(),
        [](auto x){return double(bf16_to_f32(x));});
    require(from_device<std::uint8_t>(cache.k,cache.host.k.size())==cache.host.k &&
        from_device<std::uint8_t>(cache.v,cache.host.v.size())==cache.host.v &&
        from_device<std::uint16_t>(cache.ks,cache.host.ks.size())==cache.host.ks &&
        from_device<std::uint16_t>(cache.vs,cache.host.vs.size())==cache.host.vs,
        "partial kernels changed immutable code/scale planes");
    return verify_reduction("owner small partial FP64",actual,expected,{1.0/256,1.1e-3,3.9e-3});
}

void append_case(Format format,int tokens,int offset) {
    std::cout << "[kernel-case] append format=" << name(format) << " T=" << tokens
        << " first_offset=" << offset << " fragmented=1\n" << std::flush;
    Cache cache(format);
    cache.k.fill(205); cache.v.fill(205); cache.ks.fill(205); cache.vs.fill(205);
    std::vector<std::int32_t> table{3,1,0,2}, positions(tokens);
    std::iota(positions.begin(),positions.end(),offset);
    auto dt=to_device(table), dp=to_device(positions);
    auto view=cache.view(0,4); view.block_table=Tensor(dt.p,DType::I32,{4});
    std::vector<std::uint16_t> zeros(tokens*1024,0);
    auto dk=to_device(zeros), dv=to_device(zeros);
    ops::gqa_kv_append(Tensor(dk.p,DType::BF16,{256,4,tokens}),
        Tensor(dv.p,DType::BF16,{256,4,tokens}),Tensor(dp.p,DType::I32,{tokens}),view,nullptr);
    cuda_synchronize();
    const auto k=from_device<std::uint8_t>(cache.k,cache.host.k.size()),
        v=from_device<std::uint8_t>(cache.v,cache.host.v.size());
    const auto ks=from_device<std::uint16_t>(cache.ks,cache.host.ks.size()),
        vs=from_device<std::uint16_t>(cache.vs,cache.host.vs.size());
    const int bytes=cache.host.code_dim*(format==Format::Bf16?2:1);
    for(int page=0;page<4;++page) for(int h=0;h<4;++h) for(int t=0;t<64;++t) {
        const auto logical=std::find(table.begin(),table.end(),page)-table.begin();
        const bool touched=logical*64+t>=offset && logical*64+t<offset+tokens;
        for(int d=0;d<bytes;++d) {
            const auto index=page_index(bytes,page,h,t,d);
            require(k[index]==(touched?0:205) && v[index]==(touched?0:205),
                "zero FP64 append/code byte oracle or untouched-row guard");
        }
        if(format!=Format::Bf16) for(int g=0;g<4;++g) {
            const auto index=page_index(4,page,h,t,g);
            if(touched) require(ks[index]==0 && vs[index]==0,"zero FP64 append scale byte oracle");
            else require(ks[index]==0xcdcd && vs[index]==0xcdcd,"untouched scale-row guard");
        }
    }
}
void page_staging_case(Format format) {
    std::cout << "[kernel-case] page-staging format=" << name(format) << " pages=3 ids=3,0,2\n" << std::flush;
    const bool plain=format==Format::Bf16, packed=format==Format::Rk4v4E8;
    LayoutBuilder builder;
    PagedKVPoolSpec spec{4,4,1,PagedKVPlaneOrder::PageMajor,{}};
    spec.planes={{plain?DType::BF16:DType::I8,packed?128:256,4},
                 {plain?DType::BF16:DType::I8,packed?128:256,4}};
    if(!plain) spec.planes.insert(spec.planes.end(),{{DType::FP16,4,4},{DType::FP16,4,4}});
    auto layout=plan_paged_kv_pool(builder,spec);
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p,backing.bytes},layout);
    std::vector<std::int32_t> ids{3,0,2};
    std::size_t bytes=0; for(std::size_t p=0;p<spec.planes.size();++p) bytes+=pool.page_bytes(p);
    DeviceBuffer staged(bytes*ids.size());
    std::vector<std::byte> expected;
    for(auto page:ids) for(std::size_t p=0;p<spec.planes.size();++p) {
        std::vector<std::byte> data(pool.page_bytes(p));
        for(std::size_t i=0;i<data.size();++i) data[i]=std::byte((i*37+page*19+p*71)&255);
        pool.copy_page_from_host(p,page,data.data(),nullptr);
        expected.insert(expected.end(),data.begin(),data.end());
    }
    cuda_synchronize();
    pool.gather_to_contiguous_device(ids,staged.p,nullptr); cuda_synchronize();
    require(from_device<std::byte>(staged,expected.size())==expected,"page gather independent byte oracle");
    for(std::size_t p=0;p<spec.planes.size();++p) cuda_check(cudaMemset(pool.plane(p).data,0,pool.plane(p).bytes()),"clear test planes");
    pool.scatter_from_contiguous_device(ids,staged.p,nullptr); cuda_synchronize();
    std::size_t begin=0;
    for(auto page:ids) for(std::size_t p=0;p<spec.planes.size();++p) {
        std::vector<std::byte> data(pool.page_bytes(p));
        pool.copy_page_to_host(p,page,data.data(),nullptr); cuda_synchronize();
        require(std::equal(data.begin(),data.end(),expected.begin()+begin),"page scatter independent byte oracle");
        begin+=data.size();
    }
}
void score_case(int pages,int first,int end) {
    std::cout << "[kernel-case] score D=256 H=24 K=4 M=2 L=16 P=" << pages
              << " domain=" << first << ',' << end << '\n' << std::flush;
    DeviceBuffer q(256*24*2*2), k(256*4*pages*2), logits(pages*48*4), stats(48*2*4),
        rows(48*4), scores(pages*4), status(4);
    ops::ScoreWorkspace work{Tensor(logits.p,DType::FP32,{pages,48}),
        Tensor(stats.p,DType::FP32,{2,48}),Tensor(rows.p,DType::I32,{48}),
        Tensor(scores.p,DType::FP32,{pages}),Tensor(status.p,DType::I32,{1})};
    std::vector<std::uint16_t> query(256*24*2), means(256*4*pages);
    std::vector<double> expected(pages);
    std::vector<float> previous;
    auto half_value=[](std::uint16_t x) {
        const int exponent=(x>>10)&31, fraction=x&1023;
        return std::ldexp(1.+fraction/1024.,exponent-15);
    };
    for(int repeat=0;repeat<2;++repeat) {
        std::fill(expected.begin(),expected.end(),0.);
        for(int layer=0;layer<16;++layer) {
            for(int m=0;m<2;++m) for(int h=0;h<24;++h)
                query[256*(h+24*m)]=std::uint16_t(0x3d00+(layer*11+h*7+m*3)%64);
            for(int p=0;p<pages;++p) for(int h=0;h<4;++h)
                means[256*(h+4*p)]=std::uint16_t(0x2800+(layer*13+h*7+p*3)%127);
            q.copy_from_host(query.data(),query.size()*2); k.copy_from_host(means.data(),means.size()*2);
            ops::kvmem_score_layer(Tensor(q.p,DType::BF16,{256,24,2}),
                Tensor(k.p,DType::FP16,{256,4,pages}),first,end,16,layer==0,work,nullptr);
            cuda_synchronize();
            // Independent FP64 scalar global softmax: only dim 0 is populated,
            // exactly as the owner's nonuniform-score fixture, across all layers.
            for(int m=0;m<2;++m) for(int h=0;h<24;++h) {
                std::vector<double> values(pages);
                double maximum=-INFINITY, denominator=0.;
                for(int p=first;p<end;++p) {
                    values[p]=double(bf16_to_f32(query[256*(h+24*m)]))*
                        half_value(means[256*(h/6+4*p)])/16.;
                    maximum=std::max(maximum,values[p]);
                }
                for(int p=first;p<end;++p) denominator+=std::exp(values[p]-maximum);
                for(int p=first;p<end;++p)
                    expected[p]+=std::exp(values[p]-maximum)/denominator/(16*24);
            }
        }
        require(from_device<int>(status,1)[0]==0,"finite owner score status");
        auto actual=from_device<float>(scores,pages);
        double error=0., norm=0., mass=0.;
        for(int p=0;p<pages;++p) {
            error+=std::pow(actual[p]-expected[p],2); norm+=expected[p]*expected[p]; mass+=actual[p];
        }
        require(std::sqrt(error/norm)<=1e-4 && std::abs(mass-2.)<1e-3,
            "owner exact score domain FP64 whole-vector/mass oracle");
        if(repeat) require(actual==previous,"owner score bitwise repeatability");
        previous=std::move(actual);
    }
}
} // namespace
int main() try {
    if(cuda_unavailable()) return 77;
    int failures=0;
    for(auto cfg:configurations) for(bool mixed:{false,true}) failures+=partial_case(cfg,mixed);
    for(auto format:{Format::Bf16,Format::Int8,Format::Rk4v4E8}) {
        for(int tokens:{1,2,3,4,16,31,32,63,64}) for(int offset:{0,63}) append_case(format,tokens,offset);
        // Additional exact offsets present in the owner inventory, beyond the
        // aligned and worst cross-page cases above.
        if(format==Format::Bf16) {
            for(int offset:{1,2}) append_case(format,1,offset);
            for(int offset:{1,5}) append_case(format,3,offset);
            append_case(format,4,1); append_case(format,32,1); append_case(format,64,1);
            for(int offset:{16,32,48}) append_case(format,16,offset);
        } else append_case(format,32,3);
        page_staging_case(format);
    }
    score_case(17,1,15); score_case(18,1,16); score_case(32,1,24);
    score_case(62,1,60); score_case(62,1,62); score_case(63,0,63); score_case(128,1,126);
    std::cout << (failures?"FAIL":"PASS") << " KVMem kernel-only owner configurations\n";
    return failures?1:0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
