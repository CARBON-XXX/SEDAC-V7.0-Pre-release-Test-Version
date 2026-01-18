#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 16);
  for (char c : in) {
    switch (c) {
      case '\\':
        out += "\\\\\\\\";
        break;
      case '"':
        out += "\\\\\"";
        break;
      case '\n':
        out += "\\\\n";
        break;
      case '\r':
        out += "\\\\r";
        break;
      case '\t':
        out += "\\\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

static std::string sh_single_quote(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  out += '\'';
  for (char c : s) {
    if (c == '\'') {
      out += "'\"'\"'";
    } else {
      out += c;
    }
  }
  out += '\'';
  return out;
}

static std::optional<long long> extract_int_field(const std::string& json, const std::string& key) {
  std::string needle = "\"" + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return std::nullopt;
  pos = json.find(':', pos);
  if (pos == std::string::npos) return std::nullopt;
  pos += 1;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t'))
    pos++;
  bool neg = false;
  if (pos < json.size() && json[pos] == '-') {
    neg = true;
    pos++;
  }
  long long val = 0;
  bool any = false;
  while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
    any = true;
    val = val * 10 + (json[pos] - '0');
    pos++;
  }
  if (!any) return std::nullopt;
  return neg ? -val : val;
}

struct RunResult {
  double latency_s = 0.0;
  long long completion_tokens = 0;
  long long prompt_tokens = 0;
};

static std::string popen_read_all(const std::string& cmd) {
  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) throw std::runtime_error("popen failed");
  std::string out;
  char buf[4096];
  while (true) {
    size_t n = fread(buf, 1, sizeof(buf), fp);
    if (n > 0) out.append(buf, n);
    if (n < sizeof(buf)) break;
  }
  int rc = pclose(fp);
  if (rc != 0) {
    throw std::runtime_error("command failed rc=" + std::to_string(rc) + " cmd=" + cmd + " out=" + out);
  }
  return out;
}

static std::string http_post_json(const std::string& url, const std::string& api_key, const std::string& payload_json,
                                  double timeout_s) {
  std::ostringstream cmd;
  cmd << "curl -sS --noproxy '*' ";
  cmd << " --max-time " << timeout_s;
  cmd << " -H 'Content-Type: application/json'";
  if (!api_key.empty()) {
    cmd << " -H " << sh_single_quote("Authorization: Bearer " + api_key);
  }
  cmd << " -X POST";
  cmd << " --data-raw " << sh_single_quote(payload_json);
  cmd << " " << sh_single_quote(url);
  cmd << " -w '\\nCURL_HTTP_CODE:%{http_code}\\n'";

  std::string raw = popen_read_all(cmd.str());
  std::string marker = "\nCURL_HTTP_CODE:";
  auto pos = raw.rfind(marker);
  if (pos == std::string::npos) {
    throw std::runtime_error("curl output missing http code trailer: " + raw);
  }
  std::string body = raw.substr(0, pos);
  std::string code_str = raw.substr(pos + marker.size());
  long http_code = std::strtol(code_str.c_str(), nullptr, 10);
  if (http_code < 200 || http_code >= 300) {
    throw std::runtime_error("HTTP " + std::to_string(http_code) + ": " + body);
  }
  return body;
}

static std::string join_url(const std::string& base_url, const std::string& path) {
  if (base_url.empty()) return path;
  if (base_url.back() == '/' && !path.empty() && path.front() == '/') return base_url.substr(0, base_url.size() - 1) + path;
  if (base_url.back() != '/' && !path.empty() && path.front() != '/') return base_url + "/" + path;
  return base_url + path;
}

static std::map<std::string, std::string> parse_args(int argc, char** argv) {
  std::map<std::string, std::string> out;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) continue;
    std::string k = a.substr(2);
    std::string v = "1";
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      v = argv[i + 1];
      i++;
    }
    out[k] = v;
  }
  return out;
}

static long long to_ll(const std::map<std::string, std::string>& m, const std::string& k, long long defv) {
  auto it = m.find(k);
  if (it == m.end()) return defv;
  return std::stoll(it->second);
}

static double to_d(const std::map<std::string, std::string>& m, const std::string& k, double defv) {
  auto it = m.find(k);
  if (it == m.end()) return defv;
  return std::stod(it->second);
}

static std::string to_s(const std::map<std::string, std::string>& m, const std::string& k, const std::string& defv) {
  auto it = m.find(k);
  if (it == m.end()) return defv;
  return it->second;
}

int main(int argc, char** argv) {
  auto args = parse_args(argc, argv);
  std::string base_url = to_s(args, "base-url", "http://127.0.0.1:8000/v1");
  std::string model = to_s(args, "model", "");
  if (model.empty()) {
    std::cerr << "missing --model\n";
    return 2;
  }
  std::string api_key = to_s(args, "api-key", "");
  long long max_tokens = to_ll(args, "max-tokens", 128);
  long long warmup = to_ll(args, "warmup", 1);
  long long repeat = to_ll(args, "repeat", 5);
  double timeout_s = to_d(args, "timeout-s", 180.0);
  std::string json_out = to_s(args, "json-out", "");
  double temperature = to_d(args, "temperature", 0.0);
  double top_p = to_d(args, "top-p", 1.0);
  long long seed = to_ll(args, "seed", 1);

  std::vector<std::string> prompts = {
      "The capital of France is",
      "123 * 456 =",
      "Return raw JSON with keys a=1,b=2:",
      "Write a short Python function that parses a JSON string and returns a dict.",
  };

  auto run_once = [&](const std::string& prompt) -> RunResult {
    std::ostringstream payload;
    payload << "{";
    payload << "\"model\":\"" << json_escape(model) << "\",";
    payload << "\"prompt\":\"" << json_escape(prompt) << "\",";
    payload << "\"max_tokens\":" << max_tokens << ",";
    payload << "\"temperature\":" << temperature << ",";
    payload << "\"top_p\":" << top_p << ",";
    payload << "\"seed\":" << seed << ",";
    payload << "\"stream\":false";
    payload << "}";

    auto t0 = std::chrono::steady_clock::now();
    std::string resp = http_post_json(join_url(base_url, "/completions"), api_key, payload.str(), timeout_s);
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    RunResult r;
    r.latency_s = std::max(1e-9, dt.count());
    r.completion_tokens = extract_int_field(resp, "completion_tokens").value_or(0);
    r.prompt_tokens = extract_int_field(resp, "prompt_tokens").value_or(0);
    return r;
  };

  for (long long i = 0; i < warmup; i++) {
    (void)run_once(prompts[static_cast<size_t>(i) % prompts.size()]);
  }

  std::vector<RunResult> results;
  results.reserve(static_cast<size_t>(repeat));
  for (long long i = 0; i < std::max(1LL, repeat); i++) {
    results.push_back(run_once(prompts[static_cast<size_t>(i) % prompts.size()]));
  }

  double total_latency = 0.0;
  long long total_completion = 0;
  long long total_prompt = 0;
  double min_lat = 1e100, max_lat = 0.0;
  std::vector<double> per_tps;
  per_tps.reserve(results.size());
  for (const auto& r : results) {
    total_latency += r.latency_s;
    total_completion += r.completion_tokens;
    total_prompt += r.prompt_tokens;
    min_lat = std::min(min_lat, r.latency_s);
    max_lat = std::max(max_lat, r.latency_s);
    per_tps.push_back(r.completion_tokens / std::max(1e-9, r.latency_s));
  }
  double avg_tps = (total_completion / std::max(1e-9, total_latency));

  std::ostringstream out;
  out << "{\n";
  out << "  \"params\": {\n";
  out << "    \"mode\": \"cpp_client\",\n";
  out << "    \"base_url\": \"" << json_escape(base_url) << "\",\n";
  out << "    \"model\": \"" << json_escape(model) << "\",\n";
  out << "    \"max_tokens\": " << max_tokens << ",\n";
  out << "    \"warmup\": " << warmup << ",\n";
  out << "    \"repeat\": " << repeat << ",\n";
  out << "    \"timeout_s\": " << timeout_s << "\n";
  out << "  },\n";
  out << "  \"summary_a\": {\n";
  out << "    \"runs\": " << results.size() << ",\n";
  out << "    \"throughput\": {\"avg_completion_tps\": " << avg_tps << "},\n";
  out << "    \"latency_s\": {\"min\": " << min_lat << ", \"max\": " << max_lat << "},\n";
  out << "    \"completion_tokens\": {\"sum\": " << total_completion << "},\n";
  out << "    \"prompt_tokens\": {\"sum\": " << total_prompt << "}\n";
  out << "  }\n";
  out << "}\n";

  std::string out_str = out.str();
  if (!json_out.empty()) {
    std::ofstream f(json_out);
    f << out_str;
  }
  std::cout << out_str;
  return 0;
}
