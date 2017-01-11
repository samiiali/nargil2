#include "../../include/misc/utils.hpp"

void nargil::Tokenize(const std::string &str_in,
                      std::vector<std::string> &tokens,
                      const std::string &delimiters = " ")
{
  auto lastPos = str_in.find_first_not_of(delimiters, 0);
  auto pos = str_in.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    tokens.push_back(str_in.substr(lastPos, pos - lastPos));
    lastPos = str_in.find_first_not_of(delimiters, pos);
    pos = str_in.find_first_of(delimiters, lastPos);
  }
}
