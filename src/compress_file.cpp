#include <cstdio>
#include <cassert>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>
#include <iostream>
#include <set>
#include <cstdio>

#include "chess/Position.h"
#include "chess/Uci.h"

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = (1024*KiB);
constexpr std::size_t GiB = (1024*MiB);

constexpr std::size_t suggestedChunkSize = MiB;
constexpr std::size_t maxMovelistSize = 10*KiB; // a safe upper bound
constexpr std::size_t maxChunkSize = 100*MiB; // to prevent malformed files from causing huge allocations

using namespace std::literals;

namespace nodchip
{
    // This namespace contains modified code from https://github.com/nodchip/Stockfish
    // which is released under GPL v3 license https://www.gnu.org/licenses/gpl-3.0.html

    using namespace std;

    struct StockfishMove
    {
        [[nodiscard]] static StockfishMove fromMove(Move move)
        {
            StockfishMove sfm;

            sfm.m_raw = 0;

            unsigned moveFlag = 0;
            if (move.type == MoveType::Promotion) moveFlag = 1;
            else if (move.type == MoveType::EnPassant) moveFlag = 2;
            else if (move.type == MoveType::Castle) moveFlag = 3;

            unsigned promotionIndex = 0;
            if (move.type == MoveType::Promotion)
            {
                promotionIndex = static_cast<int>(move.promotedPiece.type()) - static_cast<int>(PieceType::Knight);
            }

            sfm.m_raw |= static_cast<std::uint16_t>(moveFlag);
            sfm.m_raw <<= 2;
            sfm.m_raw |= static_cast<std::uint16_t>(promotionIndex);
            sfm.m_raw <<= 6;
            sfm.m_raw |= static_cast<int>(move.from);
            sfm.m_raw <<= 6;
            sfm.m_raw |= static_cast<int>(move.to);

            return sfm;
        }

        [[nodiscard]] Move toMove() const
        {
            const Square to = static_cast<Square>((m_raw & (0b111111 << 0) >> 0));
            const Square from = static_cast<Square>((m_raw & (0b111111 << 6)) >> 6);

            const unsigned promotionIndex = (m_raw & (0b11 << 12)) >> 12;
            const PieceType promotionType = static_cast<PieceType>(static_cast<int>(PieceType::Knight) + promotionIndex);

            const unsigned moveFlag = (m_raw & (0b11 << 14)) >> 14;
            MoveType type = MoveType::Normal;
            if (moveFlag == 1) type = MoveType::Promotion;
            else if (moveFlag == 2) type = MoveType::EnPassant;
            else if (moveFlag == 3) type = MoveType::Castle;

            if (type == MoveType::Promotion)
            {
                const Color stm = to.rank() == rank8 ? Color::White : Color::Black;
                return Move{from, to, type, Piece(promotionType, stm)};
            }
            return Move{from, to, type};
        }

    private:
        std::uint16_t m_raw;
    };
    static_assert(sizeof(StockfishMove) == sizeof(std::uint16_t));

    struct PackedSfen
    {
        uint8_t data[32];
    };

    struct PackedSfenValue
    {
        // phase
        PackedSfen sfen;

        // Evaluation value returned from Learner::search()
        int16_t score;

        // PV first move
        // Used when finding the match rate with the teacher
        StockfishMove move;

        // Trouble of the phase from the initial phase.
        uint16_t gamePly;

        // 1 if the player on this side ultimately wins the game. -1 if you are losing.
        // 0 if a draw is reached.
        // The draw is in the teacher position generation command gensfen,
        // Only write if LEARN_GENSFEN_DRAW_RESULT is enabled.
        int8_t game_result;

        // When exchanging the file that wrote the teacher aspect with other people
        //Because this structure size is not fixed, pad it so that it is 40 bytes in any environment.
        uint8_t padding;

        // 32 + 2 + 2 + 2 + 1 + 1 = 40bytes
    };
    static_assert(sizeof(PackedSfenValue) == 40);
    // Class that handles bitstream

    // useful when doing aspect encoding
    struct BitStream
    {
        // Set the memory to store the data in advance.
        // Assume that memory is cleared to 0.
        void  set_data(uint8_t* data_) { data = data_; reset(); }

        // Get the pointer passed in set_data().
        uint8_t* get_data() const { return data; }

        // Get the cursor.
        int get_cursor() const { return bit_cursor; }

        // reset the cursor
        void reset() { bit_cursor = 0; }

        // Write 1bit to the stream.
        // If b is non-zero, write out 1. If 0, write 0.
        void write_one_bit(int b)
        {
            if (b)
                data[bit_cursor / 8] |= 1 << (bit_cursor & 7);

            ++bit_cursor;
        }

        // Get 1 bit from the stream.
        int read_one_bit()
        {
            int b = (data[bit_cursor / 8] >> (bit_cursor & 7)) & 1;
            ++bit_cursor;

            return b;
        }

        // write n bits of data
        // Data shall be written out from the lower order of d.
        void write_n_bit(int d, int n)
        {
            for (int i = 0; i <n; ++i)
                write_one_bit(d & (1 << i));
        }

        // read n bits of data
        // Reverse conversion of write_n_bit().
        int read_n_bit(int n)
        {
            int result = 0;
            for (int i = 0; i < n; ++i)
                result |= read_one_bit() ? (1 << i) : 0;

            return result;
        }

    private:
        // Next bit position to read/write.
        int bit_cursor;

        // data entity
        uint8_t* data;
    };


    // Huffman coding
    // * is simplified from mini encoding to make conversion easier.
    //
    // 1 box on the board (other than NO_PIECE) = 2 to 6 bits (+ 1-bit flag + 1-bit forward and backward)
    // 1 piece of hand piece = 1-5bit (+ 1-bit flag + 1bit ahead and behind)
    //
    // empty xxxxx0 + 0 (none)
    // step xxxx01 + 2 xxxx0 + 2
    // incense xx0011 + 2 xx001 + 2
    // Katsura xx1011 + 2 xx101 + 2
    // silver xx0111 + 2 xx011 + 2
    // Gold x01111 + 1 x0111 + 1 // Gold is valid and has no flags.
    // corner 011111 + 2 01111 + 2
    // Fly 111111 + 2 11111 + 2
    //
    // Assuming all pieces are on the board,
    // Sky 81-40 pieces = 41 boxes = 41bit
    // Walk 4bit*18 pieces = 72bit
    // Incense 6bit*4 pieces = 24bit
    // Katsura 6bit*4 pieces = 24bit
    // Silver 6bit*4 pieces = 24bit
    // Gold 6bit* 4 pieces = 24bit
    // corner 8bit* 2 pieces = 16bit
    // Fly 8bit* 2 pieces = 16bit
    // -------
    // 241bit + 1bit (turn) + 7bit × 2 (King's position after) = 256bit
    //
    // When the piece on the board moves to the hand piece, the piece on the board becomes empty, so the box on the board can be expressed with 1 bit,
    // Since the hand piece can be expressed by 1 bit less than the piece on the board, the total number of bits does not change in the end.
    // Therefore, in this expression, any aspect can be expressed by this bit number.
    // It is a hand piece and no flag is required, but if you include this, the bit number of the piece on the board will be -1
    // Since the total number of bits can be fixed, we will include this as well.

    // Huffman Encoding
    //
    // Empty  xxxxxxx0
    // Pawn   xxxxx001 + 1 bit (Side to move)
    // Knight xxxxx011 + 1 bit (Side to move)
    // Bishop xxxxx101 + 1 bit (Side to move)
    // Rook   xxxxx111 + 1 bit (Side to move)

    struct HuffmanedPiece
    {
        int code; // how it will be coded
        int bits; // How many bits do you have
    };

    // NOTE: Order adjusted for this library because originally NO_PIECE had index 0
    constexpr HuffmanedPiece huffman_table[] =
    {
        {0b0001,4}, // PAWN     1
        {0b0011,4}, // KNIGHT   3
        {0b0101,4}, // BISHOP   5
        {0b0111,4}, // ROOK     7
        {0b1001,4}, // QUEEN    9
        {-1,-1},    // KING - unused
        {0b0000,1}, // NO_PIECE 0
    };

    // Class for compressing/decompressing sfen
    // sfen can be packed to 256bit (32bytes) by Huffman coding.
    // This is proven by mini. The above is Huffman coding.
    //
    // Internal format = 1-bit turn + 7-bit king position *2 + piece on board (Huffman coding) + hand piece (Huffman coding)
    // Side to move (White = 0, Black = 1) (1bit)
    // White King Position (6 bits)
    // Black King Position (6 bits)
    // Huffman Encoding of the board
    // Castling availability (1 bit x 4)
    // En passant square (1 or 1 + 6 bits)
    // Rule 50 (6 bits)
    // Game play (8 bits)
    //
    // TODO(someone): Rename SFEN to FEN.
    //
    struct SfenPacker
    {
        // Pack sfen and store in data[32].
        void pack(const Position& pos)
        {
            memset(data, 0, 32 /* 256bit */);
            stream.set_data(data);

            // turn
            // Side to move.
            stream.write_one_bit((int)(pos.sideToMove()));

            // 7-bit positions for leading and trailing balls
            // White king and black king, 6 bits for each.
            stream.write_n_bit(static_cast<int>(pos.kingSquare(Color::White)), 6);
            stream.write_n_bit(static_cast<int>(pos.kingSquare(Color::Black)), 6);

            // Write the pieces on the board other than the kings.
            for (Rank r = rank8; r >= rank1; --r)
            {
                for (File f = fileA; f <= fileH; ++f)
                {
                    Piece pc = pos.pieceAt(Square(f, r));
                    if (pc.type() == PieceType::King)
                        continue;
                    write_board_piece_to_stream(pc);
                }
            }

            // TODO(someone): Support chess960.
            auto cr = pos.castlingRights();
            stream.write_one_bit(contains(cr, CastlingRights::WhiteKingSide));
            stream.write_one_bit(contains(cr, CastlingRights::WhiteQueenSide));
            stream.write_one_bit(contains(cr, CastlingRights::BlackKingSide));
            stream.write_one_bit(contains(cr, CastlingRights::BlackQueenSide));

            if (pos.epSquare() == Square::none()) {
                stream.write_one_bit(0);
            }
            else {
                stream.write_one_bit(1);
                stream.write_n_bit(static_cast<int>(pos.epSquare()), 6);
            }

            stream.write_n_bit(pos.rule50Counter(), 6);

            stream.write_n_bit(pos.halfMove(), 8);

            assert(stream.get_cursor() <= 256);
        }

        // sfen packed by pack() (256bit = 32bytes)
        // Or sfen to decode with unpack()
        uint8_t *data; // uint8_t[32];

        BitStream stream;

        // Output the board pieces to stream.
        void write_board_piece_to_stream(Piece pc)
        {
            // piece type
            PieceType pr = pc.type();
            auto c = huffman_table[static_cast<int>(pr)];
            stream.write_n_bit(c.code, c.bits);

            if (pc == Piece::none())
                return;

            // first and second flag
            stream.write_one_bit(static_cast<int>(pc.color()));
        }

        // Read one board piece from stream
        [[nodiscard]] Piece read_board_piece_from_stream()
        {
            int pr = static_cast<int>(PieceType::None);
            int code = 0, bits = 0;
            while (true)
            {
                code |= stream.read_one_bit() << bits;
                ++bits;

                assert(bits <= 6);

                for (pr = static_cast<int>(PieceType::Pawn); pr <= static_cast<int>(PieceType::None); ++pr)
                    if (huffman_table[pr].code == code
                        && huffman_table[pr].bits == bits)
                        goto Found;
            }
        Found:;
            if (pr == static_cast<int>(PieceType::None))
                return Piece::none();

            // first and second flag
            Color c = (Color)stream.read_one_bit();

            return Piece(static_cast<PieceType>(pr), c);
        }
    };


    [[nodiscard]] Position pos_from_packed_sfen(const PackedSfen& sfen)
    {
        SfenPacker packer;
        auto& stream = packer.stream;
        stream.set_data((uint8_t*)&sfen);

        Position pos{};

        // Active color
        pos.setSideToMove((Color)stream.read_one_bit());

        // First the position of the ball
        pos.place(Piece(PieceType::King, Color::White), static_cast<Square>(stream.read_n_bit(6)));
        pos.place(Piece(PieceType::King, Color::Black), static_cast<Square>(stream.read_n_bit(6)));

        // Piece placement
        for (Rank r = rank8; r >= rank1; --r)
        {
            for (File f = fileA; f <= fileH; ++f)
            {
                auto sq = Square(f, r);

                // it seems there are already balls
                Piece pc;
                if (pos.pieceAt(sq).type() != PieceType::King)
                {
                    assert(pos.pieceAt(sq) == Piece::none());
                    pc = packer.read_board_piece_from_stream();
                }
                else
                {
                    pc = pos.pieceAt(sq);
                }

                // There may be no pieces, so skip in that case.
                if (pc == Piece::none())
                    continue;

                if (pc.type() != PieceType::King)
                {
                    pos.place(pc, sq);
                }

                if (stream.get_cursor()> 256)
                    throw std::runtime_error("Improperly encoded bin sfen");
                //assert(stream.get_cursor() <= 256);
            }
        }

        // Castling availability.
        CastlingRights cr = CastlingRights::None;
        if (stream.read_one_bit()) {
            cr |= CastlingRights::WhiteKingSide;
        }
        if (stream.read_one_bit()) {
            cr |= CastlingRights::WhiteQueenSide;
        }
        if (stream.read_one_bit()) {
            cr |= CastlingRights::BlackKingSide;
        }
        if (stream.read_one_bit()) {
            cr |= CastlingRights::BlackQueenSide;
        }
        pos.setCastlingRights(cr);

        // En passant square. Ignore if no pawn capture is possible
        if (stream.read_one_bit()) {
            Square ep_square = static_cast<Square>(stream.read_n_bit(6));
            pos.setEpSquare(ep_square);
        }

        // Halfmove clock
        pos.setRule50Counter(stream.read_n_bit(6));

        // Fullmove number
        pos.setHalfMove(stream.read_n_bit(8));

        if (stream.get_cursor()> 256)
            throw std::runtime_error("Improperly encoded bin sfen");
        //assert(stream.get_cursor() <= 256);

        return pos;
    }
}

struct CompressedTrainingDataFile
{
    struct Header
    {
        std::uint32_t chunkSize;
    };

    CompressedTrainingDataFile(std::string path, std::ios_base::openmode om = std::ios_base::app) :
        m_path(std::move(path)),
        m_file(m_path, std::ios_base::binary | std::ios_base::in | std::ios_base::out | om)
    {
    }

    void append(const char* data, std::uint32_t size)
    {
        writeChunkHeader({size});
        m_file.write(data, size);
    }

    [[nodiscard]] bool hasNextChunk()
    {
        m_file.peek();
        return !m_file.eof();
    }

    [[nodiscard]] std::vector<unsigned char> readNextChunk()
    {
        auto size = readChunkHeader().chunkSize;
        std::vector<unsigned char> data(size);
        m_file.read(reinterpret_cast<char*>(data.data()), size);
        return data;
    }

private:
    std::string m_path;
    std::fstream m_file;

    void writeChunkHeader(Header h)
    {
        unsigned char header[8];
        header[0] = 'B';
        header[1] = 'I';
        header[2] = 'N';
        header[3] = 'P';
        header[4] = h.chunkSize;
        header[5] = h.chunkSize >> 8;
        header[6] = h.chunkSize >> 16;
        header[7] = h.chunkSize >> 24;
        m_file.write(reinterpret_cast<const char*>(header), 8);
    }

    [[nodiscard]] Header readChunkHeader()
    {
        unsigned char header[8];
        m_file.read(reinterpret_cast<char*>(header), 8);
        if (header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P')
        {
            throw std::runtime_error("Invalid binpack file or chunk.");
        }

        const std::uint32_t size =
            header[4]
            | (header[5] << 8)
            | (header[6] << 16)
            | (header[7] << 24);

        if (size > maxChunkSize)
        {
            throw std::runtime_error("Chunks size larger than supported. Malformed file?");
        }

        return { size };
    }
};

[[nodiscard]] std::uint16_t signedToUnsigned(std::int16_t a)
{
    std::uint16_t r;
    std::memcpy(&r, &a, sizeof(std::uint16_t));
    if (r & 0x8000)
    {
        r ^= 0x7FFF;
    }
    r = (r << 1) | (r >> 15);
    return r;
}

[[nodiscard]] std::int16_t unsignedToSigned(std::uint16_t r)
{
    std::int16_t a;
    r = (r << 15) | (r >> 1);
    if (r & 0x8000)
    {
        r ^= 0x7FFF;
    }
    std::memcpy(&a, &r, sizeof(std::uint16_t));
    return a;
}

struct TrainingDataEntry
{
    Position pos;
    Move move;
    std::int16_t score;
    std::uint16_t ply;
    std::int16_t result;
};

[[nodiscard]] TrainingDataEntry packedSfenValueToTrainingDataEntry(const nodchip::PackedSfenValue& psv)
{
    TrainingDataEntry ret;

    ret.pos = nodchip::pos_from_packed_sfen(psv.sfen);
    ret.move = psv.move.toMove();
    ret.score = psv.score;
    ret.ply = psv.gamePly;
    ret.result = psv.game_result;

    return ret;
}

[[nodiscard]] nodchip::PackedSfenValue trainingDataEntryToPackedSfenValue(const TrainingDataEntry& plain)
{
    nodchip::PackedSfenValue ret;

    nodchip::SfenPacker sp;
    sp.data = reinterpret_cast<uint8_t*>(&ret.sfen);
    sp.pack(plain.pos);

    ret.score = plain.score;
    ret.move = nodchip::StockfishMove::fromMove(plain.move);
    ret.gamePly = plain.ply;
    ret.game_result = plain.result;
    ret.padding = 0xff; // for consistency with the .bin format.

    return ret;
}

[[nodiscard]] bool isContinuation(const TrainingDataEntry& lhs, const TrainingDataEntry& rhs)
{
    return
        lhs.result == -rhs.result
        && lhs.ply + 1 == rhs.ply
        && lhs.pos.afterMove(lhs.move) == rhs.pos;
}

struct PackedTrainingDataEntry
{
    unsigned char bytes[32];
};

[[nodiscard]] std::size_t usedBitsSafe(std::size_t value)
{
    if (value == 0) return 0;
    return util::usedBits(value - 1);
}

static constexpr std::size_t scoreVleBlockSize = 4;

struct PackedMoveScoreListReader
{
    TrainingDataEntry entry;
    std::uint16_t numPlies;
    unsigned char* movetext;

    PackedMoveScoreListReader(const TrainingDataEntry& entry, unsigned char* movetext, std::uint16_t numPlies) :
        entry(entry),
        movetext(movetext),
        numPlies(numPlies),
        m_lastScore(-entry.score)
    {

    }

    [[nodiscard]] std::uint8_t extractBitsLE8(std::size_t count)
    {
        if (count == 0) return 0;

        if (m_readBitsLeft == 0)
        {
            m_readOffset += 1;
            m_readBitsLeft = 8;
        }

        const std::uint8_t byte = movetext[m_readOffset] << (8 - m_readBitsLeft);
        std::uint8_t bits = byte >> (8 - count);

        if (count > m_readBitsLeft)
        {
            const auto spillCount = count - m_readBitsLeft;
            bits |= movetext[m_readOffset + 1] >> (8 - spillCount);

            m_readBitsLeft += 8;
            m_readOffset += 1;
        }

        m_readBitsLeft -= count;

        return bits;
    }

    [[nodiscard]] std::uint16_t extractVle16(std::size_t blockSize)
    {
        auto mask = (1 << blockSize) - 1;
        std::uint16_t v = 0;
        std::size_t offset = 0;
        for(;;)
        {
            std::uint16_t block = extractBitsLE8(blockSize + 1);
            v |= ((block & mask) << offset);
            if (!(block >> blockSize))
            {
                break;
            }

            offset += blockSize;
        }
        return v;
    }

    [[nodiscard]] TrainingDataEntry nextEntry()
    {
        entry.pos.doMove(entry.move);
        auto [move, score] = nextMoveScore(entry.pos);
        entry.move = move;
        entry.score = score;
        entry.ply += 1;
        entry.result = -entry.result;
        return entry;
    }

    [[nodiscard]] bool hasNext() const
    {
        return m_numReadPlies < numPlies;
    }

    [[nodiscard]] std::pair<Move, std::int16_t> nextMoveScore(const Position& pos)
    {
        Move move;
        std::int16_t score;

        const Color sideToMove = pos.sideToMove();
        const Bitboard ourPieces = pos.piecesBB(sideToMove);
        const Bitboard theirPieces = pos.piecesBB(!sideToMove);
        const Bitboard occupied = ourPieces | theirPieces;

        const auto pieceId = extractBitsLE8(usedBitsSafe(ourPieces.count()));
        const auto from = Square(nthSetBitIndex(ourPieces.bits(), pieceId));

        const auto pt = pos.pieceAt(from).type();
        switch (pt)
        {
        case PieceType::Pawn:
        {
            const Rank promotionRank = pos.sideToMove() == Color::White ? rank7 : rank2;
            const Rank startRank = pos.sideToMove() == Color::White ? rank2 : rank7;
            const auto forward = sideToMove == Color::White ? FlatSquareOffset(0, 1) : FlatSquareOffset(0, -1);

            const Square epSquare = pos.epSquare();

            Bitboard attackTargets = theirPieces;
            if (epSquare != Square::none())
            {
                attackTargets |= epSquare;
            }

            Bitboard destinations = bb::pawnAttacks(Bitboard::square(from), sideToMove) & attackTargets;

            const Square sqForward = from + forward;
            if (!occupied.isSet(sqForward))
            {
                destinations |= sqForward;

                const Square sqForward2 = sqForward + forward;
                if (
                    from.rank() == startRank
                    && !occupied.isSet(sqForward2)
                    )
                {
                    destinations |= sqForward2;
                }
            }

            const auto destinationsCount = destinations.count();
            if (from.rank() == promotionRank)
            {
                const auto moveId = extractBitsLE8(usedBitsSafe(destinationsCount * 4ull));
                const Piece promotedPiece = Piece(
                    fromOrdinal<PieceType>(ordinal(PieceType::Knight) + (moveId % 4ull)),
                    sideToMove
                );
                const auto to = Square(nthSetBitIndex(destinations.bits(), moveId / 4ull));

                move = Move::promotion(from, to, promotedPiece);
                break;
            }
            else
            {
                auto moveId = extractBitsLE8(usedBitsSafe(destinationsCount));
                const auto to = Square(nthSetBitIndex(destinations.bits(), moveId));
                if (to == epSquare)
                {
                    move = Move::enPassant(from, to);
                    break;
                }
                else
                {
                    move = Move::normal(from, to);
                    break;
                }
            }
        }
        case PieceType::King:
        {
            const CastlingRights ourCastlingRightsMask =
                sideToMove == Color::White
                ? CastlingRights::White
                : CastlingRights::Black;

            const CastlingRights castlingRights = pos.castlingRights();

            const Bitboard attacks = bb::pseudoAttacks<PieceType::King>(from) & ~ourPieces;
            const std::size_t attacksSize = attacks.count();
            const std::size_t numCastlings = intrin::popcount(ordinal(castlingRights & ourCastlingRightsMask));

            const auto moveId = extractBitsLE8(usedBitsSafe(attacksSize + numCastlings));

            if (moveId >= attacksSize)
            {
                const std::size_t idx = moveId - attacksSize;

                const CastleType castleType =
                    idx == 0
                    && contains(castlingRights, CastlingTraits::castlingRights[sideToMove][CastleType::Long])
                    ? CastleType::Long
                    : CastleType::Short;

                move = Move::castle(castleType, sideToMove);
                break;
            }
            else
            {
                auto to = Square(nthSetBitIndex(attacks.bits(), moveId));
                move = Move::normal(from, to);
                break;
            }
            break;
        }
        default:
        {
            const Bitboard attacks = bb::attacks(pt, from, occupied) & ~ourPieces;
            const auto moveId = extractBitsLE8(usedBitsSafe(attacks.count()));
            auto to = Square(nthSetBitIndex(attacks.bits(), moveId));
            move = Move::normal(from, to);
            break;
        }
        }

        score = m_lastScore + unsignedToSigned(extractVle16(scoreVleBlockSize));
        m_lastScore = -score;

        ++m_numReadPlies;

        return {move, score};
    }

    [[nodiscard]] std::size_t numReadBytes()
    {
        return m_readOffset + (m_readBitsLeft != 8);
    }

private:
    std::size_t m_readBitsLeft = 8;
    std::size_t m_readOffset = 0;
    std::int16_t m_lastScore = 0;
    std::uint16_t m_numReadPlies = 0;
};

struct PackedMoveScoreList
{
    std::uint16_t numPlies = 0;
    std::vector<unsigned char> movetext;

    void clear(const TrainingDataEntry& e)
    {
        numPlies = 0;
        movetext.clear();
        m_bitsLeft = 0;
        m_lastScore = -e.score;
    }

    void addBitsLE8(std::uint8_t bits, std::size_t count)
    {
        if (count == 0) return;

        if (m_bitsLeft == 0)
        {
            movetext.emplace_back(bits << (8 - count));
            m_bitsLeft = 8;
        }
        else if (count <= m_bitsLeft)
        {
            movetext.back() |= bits << (m_bitsLeft - count);
        }
        else
        {
            const auto spillCount = count - m_bitsLeft;
            movetext.back() |= bits >> spillCount;
            movetext.emplace_back(bits << (8 - spillCount));
            m_bitsLeft += 8;
        }

        m_bitsLeft -= count;
    }

    void addBitsVle16(std::uint16_t v, std::size_t blockSize)
    {
        auto mask = (1 << blockSize) - 1;
        for(;;)
        {
            std::uint8_t block = (v & mask) | ((v > mask) << blockSize);
            addBitsLE8(block, blockSize + 1);
            v >>= blockSize;
            if (v == 0) break;
        }
    }


    void addMoveScore(const Position& pos, Move move, std::int16_t score)
    {
        const Color sideToMove = pos.sideToMove();
        const Bitboard ourPieces = pos.piecesBB(sideToMove);
        const Bitboard theirPieces = pos.piecesBB(!sideToMove);
        const Bitboard occupied = ourPieces | theirPieces;

        const std::uint8_t pieceId = (pos.piecesBB(sideToMove) & bb::before(move.from)).count();
        std::size_t numMoves = 0;
        int moveId = 0;
        const auto pt = pos.pieceAt(move.from).type();
        switch (pt)
        {
        case PieceType::Pawn:
        {
            const Rank secondToLastRank = pos.sideToMove() == Color::White ? rank7 : rank2;
            const Rank startRank = pos.sideToMove() == Color::White ? rank2 : rank7;
            const auto forward = sideToMove == Color::White ? FlatSquareOffset(0, 1) : FlatSquareOffset(0, -1);

            const Square epSquare = pos.epSquare();

            Bitboard attackTargets = theirPieces;
            if (epSquare != Square::none())
            {
                attackTargets |= epSquare;
            }

            Bitboard destinations = bb::pawnAttacks(Bitboard::square(move.from), sideToMove) & attackTargets;

            const Square sqForward = move.from + forward;
            if (!occupied.isSet(sqForward))
            {
                destinations |= sqForward;

                const Square sqForward2 = sqForward + forward;
                if (
                    move.from.rank() == startRank
                    && !occupied.isSet(sqForward2)
                    )
                {
                    destinations |= sqForward2;
                }
            }

            moveId = (destinations & bb::before(move.to)).count();
            numMoves = destinations.count();
            if (move.from.rank() == secondToLastRank)
            {
                const auto promotionIndex = (ordinal(move.promotedPiece.type()) - ordinal(PieceType::Knight));
                moveId = moveId * 4 + promotionIndex;
                numMoves *= 4;
            }

            break;
        }
        case PieceType::King:
        {
            const CastlingRights ourCastlingRightsMask =
                sideToMove == Color::White
                ? CastlingRights::White
                : CastlingRights::Black;

            const CastlingRights castlingRights = pos.castlingRights();

            const Bitboard attacks = bb::pseudoAttacks<PieceType::King>(move.from) & ~ourPieces;
            const auto attacksSize = attacks.count();
            const auto numCastlingRights = intrin::popcount(ordinal(castlingRights & ourCastlingRightsMask));

            numMoves += attacksSize;
            numMoves += numCastlingRights;

            if (move.type == MoveType::Castle)
            {
                const auto longCastlingRights = CastlingTraits::castlingRights[sideToMove][CastleType::Long];

                moveId = attacksSize - 1;

                if (contains(castlingRights, longCastlingRights))
                {
                    // We have to add one no matter if it's the used one or not.
                    moveId += 1;
                }

                if (CastlingTraits::moveCastlingType(move) == CastleType::Short)
                {
                    moveId += 1;
                }
            }
            else
            {
                moveId = (attacks & bb::before(move.to)).count();
            }
            break;
        }
        default:
        {
            const Bitboard attacks = bb::attacks(pt, move.from, occupied) & ~ourPieces;

            moveId = (attacks & bb::before(move.to)).count();
            numMoves = attacks.count();
        }
        }

        const std::size_t numPieces = ourPieces.count();
        addBitsLE8(pieceId, usedBitsSafe(numPieces));
        addBitsLE8(moveId, usedBitsSafe(numMoves));

        std::uint16_t scoreDelta = signedToUnsigned(score - m_lastScore);
        addBitsVle16(scoreDelta, scoreVleBlockSize);
        m_lastScore = -score;

        ++numPlies;
    }

private:
    std::size_t m_bitsLeft = 0;
    std::int16_t m_lastScore = 0;
};


[[nodiscard]] PackedTrainingDataEntry packEntry(const TrainingDataEntry& plain)
{
    PackedTrainingDataEntry packed;

    auto compressedPos = plain.pos.compress();
    auto compressedMove = plain.move.compress();

    static_assert(sizeof(compressedPos) + sizeof(compressedMove) + 6 == sizeof(PackedTrainingDataEntry));

    std::size_t offset = 0;
    compressedPos.writeToBigEndian(packed.bytes);
    offset += sizeof(compressedPos);
    compressedMove.writeToBigEndian(packed.bytes + offset);
    offset += sizeof(compressedMove);
    std::uint16_t pr = plain.ply | (signedToUnsigned(plain.result) << 14);
    packed.bytes[offset++] = signedToUnsigned(plain.score) >> 8;
    packed.bytes[offset++] = signedToUnsigned(plain.score);
    packed.bytes[offset++] = pr >> 8;
    packed.bytes[offset++] = pr;
    packed.bytes[offset++] = plain.pos.rule50Counter() >> 8;
    packed.bytes[offset++] = plain.pos.rule50Counter();

    return packed;
}

[[nodiscard]] TrainingDataEntry unpackEntry(const PackedTrainingDataEntry& packed)
{
    TrainingDataEntry plain;

    std::size_t offset = 0;
    auto compressedPos = CompressedPosition::readFromBigEndian(packed.bytes);
    plain.pos = compressedPos.decompress();
    offset += sizeof(compressedPos);
    auto compressedMove = CompressedMove::readFromBigEndian(packed.bytes + offset);
    plain.move = compressedMove.decompress();
    offset += sizeof(compressedMove);
    plain.score = unsignedToSigned((packed.bytes[offset] << 8) | packed.bytes[offset+1]);
    offset += 2;
    std::uint16_t pr = (packed.bytes[offset] << 8) | packed.bytes[offset+1];
    plain.ply = pr & 0x3FFF;
    plain.pos.setPly(plain.ply);
    plain.result = unsignedToSigned(pr >> 14);
    offset += 2;
    plain.pos.setRule50Counter((packed.bytes[offset] << 8) | packed.bytes[offset+1]);

    return plain;
}

struct CompressedTrainingDataEntryWriter
{
    static constexpr std::size_t chunkSize = suggestedChunkSize;

    CompressedTrainingDataEntryWriter(std::string path, std::ios_base::openmode om = std::ios_base::app) :
        m_outputFile(path, om),
        m_lastEntry{},
        m_movelist{},
        m_packedSize(0),
        m_packedEntries(chunkSize + maxMovelistSize),
        m_isFirst(true)
    {
        m_lastEntry.ply = 0xFFFF; // so it's never a continuation
        m_lastEntry.result = 0x7FFF;
    }

    void addTrainingDataEntry(const TrainingDataEntry& e)
    {
        bool isCont = isContinuation(m_lastEntry, e);
        if (isCont)
        {
            // add to movelist
            m_movelist.addMoveScore(e.pos, e.move, e.score);
        }
        else
        {
            if (!m_isFirst)
            {
                writeMovelist();
            }

            if (m_packedSize >= chunkSize)
            {
                m_outputFile.append(m_packedEntries.data(), m_packedSize);
                m_packedSize = 0;
            }

            auto packed = packEntry(e);
            std::memcpy(m_packedEntries.data() + m_packedSize, &packed, sizeof(PackedTrainingDataEntry));
            m_packedSize += sizeof(PackedTrainingDataEntry);

            m_movelist.clear(e);

            m_isFirst = false;
        }

        m_lastEntry = e;
    }

    ~CompressedTrainingDataEntryWriter()
    {
        if (m_packedSize > 0)
        {
            if (!m_isFirst)
            {
                writeMovelist();
            }

            m_outputFile.append(m_packedEntries.data(), m_packedSize);
            m_packedSize = 0;
        }
    }

private:
    CompressedTrainingDataFile m_outputFile;
    TrainingDataEntry m_lastEntry;
    PackedMoveScoreList m_movelist;
    std::size_t m_packedSize;
    std::vector<char> m_packedEntries;
    bool m_isFirst;

    void writeMovelist()
    {
        m_packedEntries[m_packedSize++] = m_movelist.numPlies >> 8;
        m_packedEntries[m_packedSize++] = m_movelist.numPlies;
        if (m_movelist.numPlies > 0)
        {
            std::memcpy(m_packedEntries.data() + m_packedSize, m_movelist.movetext.data(), m_movelist.movetext.size());
            m_packedSize += m_movelist.movetext.size();
        }
    };
};

struct CompressedTrainingDataEntryReader
{
    static constexpr std::size_t chunkSize = suggestedChunkSize;

    CompressedTrainingDataEntryReader(std::string path, std::ios_base::openmode om = std::ios_base::app) :
        m_inputFile(path, om),
        m_chunk(),
        m_movelistReader(std::nullopt),
        m_offset(0),
        m_isEnd(false)
    {
        if (!m_inputFile.hasNextChunk())
        {
            m_isEnd = true;
        }
        else
        {
            m_chunk = m_inputFile.readNextChunk();
        }
    }

    [[nodiscard]] bool hasNext()
    {
        return !m_isEnd;
    }

    [[nodiscard]] TrainingDataEntry next()
    {
        if (m_movelistReader.has_value())
        {
            const auto e = m_movelistReader->nextEntry();

            if (!m_movelistReader->hasNext())
            {
                m_offset += m_movelistReader->numReadBytes();
                m_movelistReader.reset();

                fetchNextChunkIfNeeded();
            }

            return e;
        }

        PackedTrainingDataEntry packed;
        std::memcpy(&packed, m_chunk.data() + m_offset, sizeof(PackedTrainingDataEntry));
        m_offset += sizeof(PackedTrainingDataEntry);

        const std::uint16_t numPlies = (m_chunk[m_offset] << 8) | m_chunk[m_offset + 1];
        m_offset += 2;

        const auto e = unpackEntry(packed);

        if (numPlies > 0)
        {
            m_movelistReader.emplace(e, reinterpret_cast<unsigned char*>(m_chunk.data()) + m_offset, numPlies);
        }
        else
        {
            fetchNextChunkIfNeeded();
        }

        return e;
    }

private:
    CompressedTrainingDataFile m_inputFile;
    std::vector<unsigned char> m_chunk;
    std::optional<PackedMoveScoreListReader> m_movelistReader;
    std::size_t m_offset;
    bool m_isEnd;

    void fetchNextChunkIfNeeded()
    {
        if (m_offset + sizeof(PackedTrainingDataEntry) + 2 > m_chunk.size())
        {
            if (m_inputFile.hasNextChunk())
            {
                m_chunk = m_inputFile.readNextChunk();
                m_offset = 0;
            }
            else
            {
                m_isEnd = true;
            }
        }
    }
};

void emitPlainEntry(std::string& buffer, const TrainingDataEntry& plain)
{
    buffer += "fen ";
    buffer += plain.pos.fen();
    buffer += '\n';

    buffer += "move ";
    buffer += uci::moveToUci(plain.pos, plain.move);
    buffer += '\n';

    buffer += "score ";
    buffer += std::to_string(plain.score);
    buffer += '\n';

    buffer += "ply ";
    buffer += std::to_string(plain.ply);
    buffer += '\n';

    buffer += "result ";
    buffer += std::to_string(plain.result);
    buffer += "\ne\n";
}

void emitBinEntry(std::vector<char>& buffer, const TrainingDataEntry& plain)
{
    auto psv = trainingDataEntryToPackedSfenValue(plain);
    const char* data = reinterpret_cast<const char*>(&psv);
    buffer.insert(buffer.end(), data, data+sizeof(psv));
}

void compressPlain(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    constexpr std::size_t reportEveryNPositions = 100'000;
    constexpr std::size_t chunkSize = suggestedChunkSize;

    std::cout << "Compressing " << inputPath << " to " << outputPath << '\n';

    CompressedTrainingDataEntryWriter writer(outputPath, om);
    TrainingDataEntry e;

    std::string key;
    std::string value;
    std::string move;

    std::ifstream inputFile(inputPath);
    const auto base = inputFile.tellg();
    std::size_t numProcessedPositions = 0;

    for(;;)
    {
        inputFile >> key;
        if (!inputFile)
        {
            break;
        }

        if (key == "e"sv)
        {
            e.move = uci::uciToMove(e.pos, move);

            writer.addTrainingDataEntry(e);

            ++numProcessedPositions;
            const auto cur = inputFile.tellg();
            if (numProcessedPositions % reportEveryNPositions == 0)
            {
                std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
            }

            continue;
        }

        inputFile >> std::ws;
        std::getline(inputFile, value, '\n');

        if (key == "fen"sv) e.pos = Position::fromFen(value.c_str());
        if (key == "move"sv) move = value;
        if (key == "score"sv) e.score = std::stoi(value);
        if (key == "ply"sv) e.ply = std::stoi(value);
        if (key == "result"sv) e.result = std::stoi(value);
    }
}

void decompressPlain(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    constexpr std::size_t bufferSize = MiB;

    std::cout << "Decompressing " << inputPath << " to " << outputPath << '\n';

    CompressedTrainingDataEntryReader reader(inputPath);
    std::ofstream outputFile(outputPath, om);
    const auto base = outputFile.tellp();
    std::size_t numProcessedPositions = 0;
    std::string buffer;
    buffer.reserve(bufferSize * 2);

    while(reader.hasNext())
    {
        emitPlainEntry(buffer, reader.next());

        ++numProcessedPositions;

        if (buffer.size() > bufferSize)
        {
            outputFile << buffer;
            buffer.clear();

            const auto cur = outputFile.tellp();
            std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
        }
    }

    if (!buffer.empty())
    {
        outputFile << buffer;

        const auto cur = outputFile.tellp();
        std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
    }
}


void compressBin(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    constexpr std::size_t reportEveryNPositions = 100'000;
    constexpr std::size_t chunkSize = suggestedChunkSize;

    std::cout << "Compressing " << inputPath << " to " << outputPath << '\n';

    CompressedTrainingDataEntryWriter writer(outputPath, om);
    TrainingDataEntry e;

    std::string key;
    std::string value;
    std::string move;

    std::ifstream inputFile(inputPath, std::ios_base::binary);
    const auto base = inputFile.tellg();
    std::size_t numProcessedPositions = 0;

    nodchip::PackedSfenValue psv;
    for(;;)
    {
        inputFile.read(reinterpret_cast<char*>(&psv), sizeof(psv));
        if (inputFile.gcount() != 40)
        {
            break;
        }

        writer.addTrainingDataEntry(packedSfenValueToTrainingDataEntry(psv));

        ++numProcessedPositions;
        const auto cur = inputFile.tellg();
        if (numProcessedPositions % reportEveryNPositions == 0)
        {
            std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
        }
    }
}

void decompressBin(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    constexpr std::size_t bufferSize = MiB;

    std::cout << "Decompressing " << inputPath << " to " << outputPath << '\n';

    CompressedTrainingDataEntryReader reader(inputPath);
    std::ofstream outputFile(outputPath, std::ios_base::binary | om);
    const auto base = outputFile.tellp();
    std::size_t numProcessedPositions = 0;
    std::vector<char> buffer;
    buffer.reserve(bufferSize * 2);

    while(reader.hasNext())
    {
        emitBinEntry(buffer, reader.next());

        ++numProcessedPositions;

        if (buffer.size() > bufferSize)
        {
            outputFile.write(buffer.data(), buffer.size());
            buffer.clear();

            const auto cur = outputFile.tellp();
            std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
        }
    }

    if (!buffer.empty())
    {
        outputFile.write(buffer.data(), buffer.size());

        const auto cur = outputFile.tellp();
        std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
    }
}

void convertBinToPlain(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    constexpr std::size_t reportEveryNPositions = 100'000;
    constexpr std::size_t bufferSize = MiB;

    std::cout << "Converting " << inputPath << " to " << outputPath << '\n';

    TrainingDataEntry e;

    std::string key;
    std::string value;
    std::string move;

    std::ifstream inputFile(inputPath, std::ios_base::binary);
    const auto base = inputFile.tellg();
    std::size_t numProcessedPositions = 0;

    std::ofstream outputFile(outputPath, om);
    std::string buffer;
    buffer.reserve(bufferSize * 2);

    nodchip::PackedSfenValue psv;
    for(;;)
    {
        inputFile.read(reinterpret_cast<char*>(&psv), sizeof(psv));
        if (inputFile.gcount() != 40)
        {
            break;
        }

        emitPlainEntry(buffer, packedSfenValueToTrainingDataEntry(psv));

        ++numProcessedPositions;

        if (buffer.size() > bufferSize)
        {
            outputFile << buffer;
            buffer.clear();

            const auto cur = outputFile.tellp();
            std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
        }
    }

    if (!buffer.empty())
    {
        outputFile << buffer;

        const auto cur = outputFile.tellp();
        std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
    }
}

void convertPlainToBin(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    constexpr std::size_t reportEveryNPositions = 100'000;
    constexpr std::size_t bufferSize = MiB;

    std::cout << "Compressing " << inputPath << " to " << outputPath << '\n';

    std::ofstream outputFile(outputPath, std::ios_base::binary | om);
    std::vector<char> buffer;
    buffer.reserve(bufferSize * 2);

    TrainingDataEntry e;

    std::string key;
    std::string value;
    std::string move;

    std::ifstream inputFile(inputPath);
    const auto base = inputFile.tellg();
    std::size_t numProcessedPositions = 0;

    for(;;)
    {
        inputFile >> key;
        if (!inputFile)
        {
            break;
        }

        if (key == "e"sv)
        {
            e.move = uci::uciToMove(e.pos, move);

            emitBinEntry(buffer, e);

            ++numProcessedPositions;

            if (buffer.size() > bufferSize)
            {
                outputFile.write(buffer.data(), buffer.size());
                buffer.clear();

                const auto cur = outputFile.tellp();
                std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
            }

            continue;
        }

        inputFile >> std::ws;
        std::getline(inputFile, value, '\n');

        if (key == "fen"sv) e.pos = Position::fromFen(value.c_str());
        if (key == "move"sv) move = value;
        if (key == "score"sv) e.score = std::stoi(value);
        if (key == "ply"sv) e.ply = std::stoi(value);
        if (key == "result"sv) e.result = std::stoi(value);
    }

    if (!buffer.empty())
    {
        outputFile.write(buffer.data(), buffer.size());

        const auto cur = outputFile.tellp();
        std::cout << "Processed " << (cur - base) << " bytes and " << numProcessedPositions << " positions.\n";
    }
}

const std::string plainExtension = ".plain";
const std::string binExtension = ".bin";
const std::string binpackExtension = ".binpack";

bool fileExists(const std::string& name)
{
    std::ifstream f(name);
    return f.good();
}

bool endsWith(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

void compress(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    if (!endsWith(outputPath, binpackExtension))
    {
        outputPath += binpackExtension;
    }

    if (endsWith(inputPath, binExtension))
    {
        compressBin(inputPath, outputPath, om);
    }
    else if (endsWith(inputPath, plainExtension))
    {
        compressPlain(inputPath, outputPath, om);
    }
    else
    {
        std::cerr << "Unrecognized file format. Only " << binExtension << " and " << plainExtension << " are supported for compression.";
    }
}

void decompress(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    if (!endsWith(inputPath, binpackExtension))
    {
        std::cerr << "Only " << binpackExtension << " files can be decompressed.\n";
        return;
    }

    if (endsWith(outputPath, binExtension))
    {
        decompressBin(inputPath, outputPath, om);
    }
    else if (endsWith(outputPath, plainExtension))
    {
        decompressPlain(inputPath, outputPath, om);
    }
    else
    {
        std::cerr << "Unrecognized file format. Only " << binExtension << " and " << plainExtension << " are supported for decompression.";
    }
}

void convert(std::string inputPath, std::string outputPath, std::ios_base::openmode om)
{
    if (!fileExists(inputPath))
    {
        std::cerr << "Input file doesn't exist.\n";
        return;
    }

    if (endsWith(inputPath, binExtension) && endsWith(outputPath, plainExtension))
    {
        convertBinToPlain(inputPath, outputPath, om);
    }
    else if (endsWith(inputPath, plainExtension) && endsWith(outputPath, binExtension))
    {
        convertPlainToBin(inputPath, outputPath, om);
    }
    else if (endsWith(inputPath, plainExtension) || endsWith(inputPath, binExtension))
    {
        compress(inputPath, outputPath, om);
    }
    else if (endsWith(inputPath, binpackExtension))
    {
        decompress(inputPath, outputPath, om);
    }
    else
    {
        std::cerr << "Unsupported extension.";
    }
}

void help()
{
    std::cout << "Usage:\n";
    std::cout << "    nnue_data_compression [-h] [-a] input_path output_path\n";
    std::cout << "\n";
    std::cout << "-h, --help                show help\n";
    std::cout << "-a, --append              append to the output file instead of truncating it\n";
    std::cout << "\n";
    std::cout << "input_path                the path to the file to process\n";
    std::cout << "output_path               the path to the file to create/append to\n";
    std::cout << "\n";
    std::cout << "Behaviour depends on file extensions. If the input\n";
    std::cout << "file has extension either " << binExtension << " or " << plainExtension << "\n";
    std::cout << "it will be compressed. The output file has then an implied\n";
    std::cout << "extension of " << binpackExtension << " and it doesn't have to be specified.\n";
    std::cout << "If the input file's extension is " << binpackExtension << " then it will be decompressed\n";
    std::cout << "to either a " << binExtension << " or " << plainExtension << " file, depending on the extension.\n";
    std::cout << "\n";
    std::cout << "Example usage:\n";
    std::cout << "1. convert from plain to binpack in append mode:\n";
    std::cout << "    nnue_data_compression -a data.plain data\n";
    std::cout << "2. convert from binpack to plain in truncate/replace mode:\n";
    std::cout << "    nnue_data_compression data.binpack data.plain\n";
}

[[nodiscard]] int run(
    const std::string& programName,
    const std::set<std::string>& flag,
    const std::vector<std::string>& pos)
{
    if (pos.size() == 0 || flag.count("help") == 1 || flag.count("h") == 1)
    {
        help();
        return 0;
    }

    if (pos.size() == 2)
    {
        std::ios_base::openmode om = std::ios_base::trunc;

        if (flag.count("a") == 1 || flag.count("append") == 1)
        {
            om = std::ios_base::app;
        }

        convert(pos[0], pos[1], om);
        return 0;
    }

    std::cerr << "Invalid arguments.\n";
    help();
    return 1;
}

[[nodiscard]] std::pair<std::set<std::string>, std::vector<std::string>> readArgs(int argc, char** argv)
{
    std::set<std::string> flags;
    std::vector<std::string> pos;

    for(int i = 1; i < argc; ++i)
    {
        if (*argv[i] == '-')
        {
            flags.emplace(argv[i]+1);
        }
        else
        {
            pos.emplace_back(argv[i]);
        }
    }

    return {flags, pos};
}

int main(int argc, char** argv)
{
    try
    {
        auto [flag, pos] = readArgs(argc, argv);
        return run(argv[0], flag, pos);
    }
    catch(std::runtime_error& e)
    {
        std::cerr << e.what() << "\n";
        std::cerr << "Exiting...\n";
    }
}
