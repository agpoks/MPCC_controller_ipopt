#pragma once
// Minimal flat-key YAML loader.
// Supports:  key: value   key: 1.234   # comment lines   blank lines
// Does NOT support nested maps, lists, or multi-line values.
#include <map>
#include <string>
#include <fstream>
#include <stdexcept>

inline std::map<std::string, std::string> load_yaml_flat(const std::string& path)
{
    std::map<std::string, std::string> result;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open YAML file: " + path);

    auto trim = [](std::string s) -> std::string {
        const std::string ws = " \t\r\n";
        auto a = s.find_first_not_of(ws);
        if (a == std::string::npos) return {};
        auto b = s.find_last_not_of(ws);
        return s.substr(a, b - a + 1);
    };

    std::string line;
    while (std::getline(f, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));
        if (!key.empty() && !val.empty()) result[key] = val;
    }
    return result;
}

inline double yaml_double(const std::map<std::string, std::string>& m,
                          const std::string& key, double def)
{
    auto it = m.find(key);
    if (it == m.end()) return def;
    try { return std::stod(it->second); } catch (...) { return def; }
}

inline int yaml_int(const std::map<std::string, std::string>& m,
                    const std::string& key, int def)
{
    auto it = m.find(key);
    if (it == m.end()) return def;
    try { return std::stoi(it->second); } catch (...) { return def; }
}

inline bool yaml_bool(const std::map<std::string, std::string>& m,
                      const std::string& key, bool def)
{
    auto it = m.find(key);
    if (it == m.end()) return def;
    const auto& v = it->second;
    return v == "true" || v == "True" || v == "1" || v == "yes";
}
