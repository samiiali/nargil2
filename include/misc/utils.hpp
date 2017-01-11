#include <string>
#include <vector>

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

namespace nargil
{
/**
 * This is a simple tokenizer function. I could not find a better
 * place to put it.
 */
void Tokenize(const std::string &str_in,
              std::vector<std::string> &tokens,
              const std::string &delimiters);
}

#include "../../source/misc/utils.cpp"

#endif
