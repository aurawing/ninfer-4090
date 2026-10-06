#include "product/prompt_input/prompt_input.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
    using Json = nlohmann::json;
    const auto path = std::filesystem::temp_directory_path() / "ninfer-product-current-input-test.json";
    const auto read = [&](const Json& root) {
        { std::ofstream stream(path); stream << root.dump(); }
        return ninfer::product::prompt_from_messages(path, false, false);
    };
    try {
        const auto messages = Json::array({{{"role", "system"}, {"content", "instruction"}},
            {{"role", "user"}, {"content", "prior user"}},
            {{"role", "tool"}, {"content", "current tool"}},
            {{"role", "user"}, {"content", "current user"}}});
        const auto legacy = read(messages);
        const auto explicit_event = read({{"messages", messages}, {"current_input_message", 2}});
        if (legacy.current_input_message || explicit_event.current_input_message != 2 ||
            legacy.messages.size() != explicit_event.messages.size())
            throw std::runtime_error("product boundary was lost or legacy changed");
        for (std::size_t i = 0; i < legacy.messages.size(); ++i)
            if (legacy.messages[i].role != explicit_event.messages[i].role ||
                legacy.messages[i].parts[0].text != explicit_event.messages[i].parts[0].text)
                throw std::runtime_error("event metadata changed message rendering source");
        for (const auto& bad : {Json(true), Json(false), Json(-1), Json(0.5), Json(2.0),
                              Json(4), Json(0), Json("2"), Json(nullptr), Json(UINT64_MAX)}) {
            bool rejected = false;
            try { (void)read({{"messages", messages}, {"current_input_message", bad}}); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid current_input_message accepted: " + bad.dump());
        }
        std::filesystem::remove(path);
        std::cout << "product current input metadata PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
