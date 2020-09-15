# [deprecated]

This tool is now embedded in the [learn branch of stockfish](https://github.com/nodchip/Stockfish). Please use `stockfish.exe convert fromfile tofile [app]` instead. Any further development will take place in the stockfish repository. The tool in this repository will not be updated.

# NNUE Data Compressor

NNUE Data Compressor is a tool to maximally compress NNUE training data for chess. It supports .plain and .bin formats. Functions effectively like a converter, i.e. it's possible to compress .plain but decompress to .bin and vice versa. On .plain files it achieves compression ratio between 40x to 50x which translates to ~15-20x on .bin files.

For example usage see "--help".


# Building
Requires a compiler with support for C++17.

`make release`
