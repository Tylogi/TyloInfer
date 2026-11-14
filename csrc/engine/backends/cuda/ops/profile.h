#pragma once

#include <string>
#include <utility>
#include <vector>

namespace tyloi::cuda {
class DecodeWindow;
void append_graph_profile_metrics(const DecodeWindow& window, const std::string& prefix,
                                  std::vector<std::pair<std::string, double>>& result);
}
