#include <iostream>
#include <string>
#include <fstream>
#include "bpe.h"

bool BPEDecoder::load(const std::string &vocab_path)
{
    std::ifstream file(vocab_path, std::ios::binary);
    if (!file)
        return false;

    vocab.clear();

    while (true)
    {
        unsigned char len;
        if (!file.read((char *)&len, 1))
            break;

        std::string token(len, '\0'); // token has '\0' * len times
        if (!file.read(&token[0], len))
            break;

        vocab.push_back(token);
    }

    return true;
}

std::string BPEDecoder::decode(const std::vector<int> &tokens)
{
    std::string output;

    for (int tok : tokens)
    {
        if (tok == -1)
        {
            output += "(?)"; // possible unknown token
            continue;
        }

        if (tok < 0 || tok >= vocab.size())
            break;

        output += vocab[tok];
    }

    return output;
}