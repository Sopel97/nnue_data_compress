#pragma once

#include "detail/ParserBits.h"

#include "Bitboard.h"
#include "CastlingTraits.h"
#include "Chess.h"
#include "Zobrist.h"

#include "enum/EnumArray.h"

#include "util/Assert.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>

struct Board
{
    constexpr Board() noexcept :
        m_pieces{},
        m_pieceBB{},
        m_piecesByColorBB{},
        m_pieceCount{}
    {
        m_pieces.fill(Piece::none());
        m_pieceBB.fill(Bitboard::none());
        m_pieceBB[Piece::none()] = Bitboard::all();
        m_piecesByColorBB.fill(Bitboard::none());
        m_pieceCount.fill(0);
        m_pieceCount[Piece::none()] = 64;
    }

    [[nodiscard]] INTRIN_CONSTEXPR bool isValid() const
    {
        if (piecesBB(whiteKing).count() != 1) return false;
        if (piecesBB(blackKing).count() != 1) return false;
        if (((piecesBB(whitePawn) | piecesBB(blackPawn)) & (bb::rank(rank1) | bb::rank(rank8))).any()) return false;
        return true;
    }

    [[nodiscard]] std::string fen() const;

    [[nodiscard]] INTRIN_CONSTEXPR bool trySet(std::string_view boardState)
    {
        File f = fileA;
        Rank r = rank8;
        bool lastWasSkip = false;
        for (auto c : boardState)
        {
            Piece piece = Piece::none();
            switch (c)
            {
            case 'r':
                piece = Piece(PieceType::Rook, Color::Black);
                break;
            case 'n':
                piece = Piece(PieceType::Knight, Color::Black);
                break;
            case 'b':
                piece = Piece(PieceType::Bishop, Color::Black);
                break;
            case 'q':
                piece = Piece(PieceType::Queen, Color::Black);
                break;
            case 'k':
                piece = Piece(PieceType::King, Color::Black);
                break;
            case 'p':
                piece = Piece(PieceType::Pawn, Color::Black);
                break;

            case 'R':
                piece = Piece(PieceType::Rook, Color::White);
                break;
            case 'N':
                piece = Piece(PieceType::Knight, Color::White);
                break;
            case 'B':
                piece = Piece(PieceType::Bishop, Color::White);
                break;
            case 'Q':
                piece = Piece(PieceType::Queen, Color::White);
                break;
            case 'K':
                piece = Piece(PieceType::King, Color::White);
                break;
            case 'P':
                piece = Piece(PieceType::Pawn, Color::White);
                break;

            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            {
                if (lastWasSkip) return false;
                lastWasSkip = true;

                const int skip = c - '0';
                f += skip;
                if (f > fileH + 1) return false;
                break;
            }

            case '/':
                lastWasSkip = false;
                if (f != fileH + 1) return false;
                f = fileA;
                --r;
                break;

            default:
                return false;
            }

            if (piece != Piece::none())
            {
                lastWasSkip = false;

                const Square sq(f, r);
                if (!sq.isOk()) return false;

                place(piece, sq);
                ++f;
            }
        }

        if (f != fileH + 1) return false;
        if (r != rank1) return false;

        return isValid();
    }

    // returns side to move
    [[nodiscard]] constexpr const char* set(const char* fen)
    {
        ASSERT(fen != nullptr);

        File f = fileA;
        Rank r = rank8;
        auto current = fen;
        bool done = false;
        while (*current != '\0')
        {
            Piece piece = Piece::none();
            switch (*current)
            {
            case 'r':
                piece = Piece(PieceType::Rook, Color::Black);
                break;
            case 'n':
                piece = Piece(PieceType::Knight, Color::Black);
                break;
            case 'b':
                piece = Piece(PieceType::Bishop, Color::Black);
                break;
            case 'q':
                piece = Piece(PieceType::Queen, Color::Black);
                break;
            case 'k':
                piece = Piece(PieceType::King, Color::Black);
                break;
            case 'p':
                piece = Piece(PieceType::Pawn, Color::Black);
                break;

            case 'R':
                piece = Piece(PieceType::Rook, Color::White);
                break;
            case 'N':
                piece = Piece(PieceType::Knight, Color::White);
                break;
            case 'B':
                piece = Piece(PieceType::Bishop, Color::White);
                break;
            case 'Q':
                piece = Piece(PieceType::Queen, Color::White);
                break;
            case 'K':
                piece = Piece(PieceType::King, Color::White);
                break;
            case 'P':
                piece = Piece(PieceType::Pawn, Color::White);
                break;

            case ' ':
                done = true;
                break;

            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            {
                const int skip = (*current) - '0';
                f += skip;
                break;
            }

            case '/':
                f = fileA;
                --r;
                break;

            default:
                break;
            }

            if (done)
            {
                break;
            }

            if (piece != Piece::none())
            {
                place(piece, Square(f, r));
                ++f;
            }

            ++current;
        }

        return current;
    }

    static constexpr Board fromFen(const char* fen)
    {
        Board board;
        (void)board.set(fen);
        return board;
    }

    [[nodiscard]] constexpr friend bool operator==(const Board& lhs, const Board& rhs) noexcept
    {
        bool equal = true;
        for (Square sq = a1; sq <= h8; ++sq)
        {
            if (lhs.m_pieces[sq] != rhs.m_pieces[sq])
            {
                equal = false;
                break;
            }
        }

        ASSERT(bbsEqual(lhs, rhs) == equal);

        return equal;
    }

    constexpr void place(Piece piece, Square sq)
    {
        ASSERT(sq.isOk());

        auto oldPiece = m_pieces[sq];
        m_pieceBB[oldPiece] ^= sq;
        if (oldPiece != Piece::none())
        {
            m_piecesByColorBB[oldPiece.color()] ^= sq;
        }
        m_pieces[sq] = piece;
        m_pieceBB[piece] |= sq;
        m_piecesByColorBB[piece.color()] |= sq;
        --m_pieceCount[oldPiece];
        ++m_pieceCount[piece];
    }

    constexpr void place(Piece piece, Square sq, ZobristKey& zobrist)
    {
        ASSERT(sq.isOk());

        auto oldPiece = m_pieces[sq];
        m_pieceBB[oldPiece] ^= sq;

        zobrist ^= Zobrist::psq[oldPiece][sq];
        zobrist ^= Zobrist::psq[piece][sq];

        if (oldPiece != Piece::none())
        {
            m_piecesByColorBB[oldPiece.color()] ^= sq;
        }
        m_pieces[sq] = piece;
        m_pieceBB[piece] |= sq;
        m_piecesByColorBB[piece.color()] |= sq;
        --m_pieceCount[oldPiece];
        ++m_pieceCount[piece];
    }

    // returns captured piece
    // doesn't check validity
    FORCEINLINE constexpr Piece doMove(Move move)
    {
        if (move.type == MoveType::Normal)
        {
            const Piece capturedPiece = m_pieces[move.to];
            const Piece piece = m_pieces[move.from];

            const Bitboard frombb = Bitboard::square(move.from);
            const Bitboard tobb = Bitboard::square(move.to);
            const Bitboard xormove = frombb ^ tobb;

            m_pieces[move.to] = piece;
            m_pieces[move.from] = Piece::none();

            m_pieceBB[piece] ^= xormove;

            m_piecesByColorBB[piece.color()] ^= xormove;

            if (capturedPiece == Piece::none())
            {
                m_pieceBB[Piece::none()] ^= xormove;
            }
            else
            {
                m_pieceBB[capturedPiece] ^= tobb;
                m_pieceBB[Piece::none()] ^= frombb;

                m_piecesByColorBB[capturedPiece.color()] ^= tobb;

                --m_pieceCount[capturedPiece];
                ++m_pieceCount[Piece::none()];
            }

            return capturedPiece;
        }

        return doMoveColdPath(move);
    }

    NOINLINE constexpr Piece doMoveColdPath(Move move)
    {
        if (move.type == MoveType::Promotion)
        {
            // We split it even though it's similar just because
            // the normal case is much more common.
            const Piece capturedPiece = m_pieces[move.to];
            const Piece fromPiece = m_pieces[move.from];
            const Piece toPiece = move.promotedPiece;

            m_pieces[move.to] = toPiece;
            m_pieces[move.from] = Piece::none();

            m_pieceBB[fromPiece] ^= move.from;
            m_pieceBB[toPiece] ^= move.to;

            m_pieceBB[capturedPiece] ^= move.to;
            m_pieceBB[Piece::none()] ^= move.from;

            m_piecesByColorBB[fromPiece.color()] ^= move.to;
            m_piecesByColorBB[fromPiece.color()] ^= move.from;
            if (capturedPiece != Piece::none())
            {
                m_piecesByColorBB[capturedPiece.color()] ^= move.to;
                --m_pieceCount[capturedPiece];
                ++m_pieceCount[Piece::none()];
            }

            --m_pieceCount[fromPiece];
            ++m_pieceCount[toPiece];

            return capturedPiece;
        }
        else if (move.type == MoveType::EnPassant)
        {
            const Piece movedPiece = m_pieces[move.from];
            const Piece capturedPiece(PieceType::Pawn, !movedPiece.color());
            const Square capturedPieceSq(move.to.file(), move.from.rank());

            // on ep move there are 3 squares involved
            m_pieces[move.to] = movedPiece;
            m_pieces[move.from] = Piece::none();
            m_pieces[capturedPieceSq] = Piece::none();

            m_pieceBB[movedPiece] ^= move.from;
            m_pieceBB[movedPiece] ^= move.to;

            m_pieceBB[Piece::none()] ^= move.from;
            m_pieceBB[Piece::none()] ^= move.to;

            m_pieceBB[capturedPiece] ^= capturedPieceSq;
            m_pieceBB[Piece::none()] ^= capturedPieceSq;

            m_piecesByColorBB[movedPiece.color()] ^= move.to;
            m_piecesByColorBB[movedPiece.color()] ^= move.from;
            m_piecesByColorBB[capturedPiece.color()] ^= capturedPieceSq;

            --m_pieceCount[capturedPiece];
            ++m_pieceCount[Piece::none()];

            return capturedPiece;
        }
        else // if (move.type == MoveType::Castle)
        {
            const Square rookFromSq = move.to;
            const Square kingFromSq = move.from;

            const Piece rook = m_pieces[rookFromSq];
            const Piece king = m_pieces[kingFromSq];
            const Color color = king.color();

            const CastleType castleType = CastlingTraits::moveCastlingType(move);
            const Square rookToSq = CastlingTraits::rookDestination[color][castleType];
            const Square kingToSq = CastlingTraits::kingDestination[color][castleType];

            // 4 squares are involved
            m_pieces[rookFromSq] = Piece::none();
            m_pieces[kingFromSq] = Piece::none();
            m_pieces[rookToSq] = rook;
            m_pieces[kingToSq] = king;

            m_pieceBB[rook] ^= rookFromSq;
            m_pieceBB[rook] ^= rookToSq;

            m_pieceBB[king] ^= kingFromSq;
            m_pieceBB[king] ^= kingToSq;

            m_pieceBB[Piece::none()] ^= rookFromSq;
            m_pieceBB[Piece::none()] ^= rookToSq;

            m_pieceBB[Piece::none()] ^= kingFromSq;
            m_pieceBB[Piece::none()] ^= kingToSq;

            m_piecesByColorBB[color] ^= rookFromSq;
            m_piecesByColorBB[color] ^= rookToSq;
            m_piecesByColorBB[color] ^= kingFromSq;
            m_piecesByColorBB[color] ^= kingToSq;

            return Piece::none();
        }
    }

    // returns captured piece
    // doesn't check validity
    FORCEINLINE constexpr Piece doMove(Move move, ZobristKey& zobrist)
    {
        if (move.type == MoveType::Normal)
        {
            const Piece capturedPiece = m_pieces[move.to];
            const Piece piece = m_pieces[move.from];

            const Bitboard frombb = Bitboard::square(move.from);
            const Bitboard tobb = Bitboard::square(move.to);
            const Bitboard xormove = frombb ^ tobb;

            m_pieces[move.to] = piece;
            m_pieces[move.from] = Piece::none();

            zobrist ^= Zobrist::psq[piece][move.from] ^ Zobrist::psq[piece][move.to];

            m_pieceBB[piece] ^= xormove;

            m_piecesByColorBB[piece.color()] ^= xormove;

            if (capturedPiece == Piece::none())
            {
                m_pieceBB[Piece::none()] ^= xormove;
            }
            else
            {
                m_pieceBB[capturedPiece] ^= tobb;
                m_pieceBB[Piece::none()] ^= frombb;

                m_piecesByColorBB[capturedPiece.color()] ^= tobb;

                --m_pieceCount[capturedPiece];
                ++m_pieceCount[Piece::none()];

                zobrist ^= Zobrist::psq[capturedPiece][move.to];
            }

            return capturedPiece;
        }

        return doMoveColdPath(move, zobrist);
    }

    NOINLINE constexpr Piece doMoveColdPath(Move move, ZobristKey& zobrist)
    {
        if (move.type == MoveType::Promotion)
        {
            // We split it even though it's similar just because
            // the normal case is much more common.
            const Piece capturedPiece = m_pieces[move.to];
            const Piece fromPiece = m_pieces[move.from];
            const Piece toPiece = move.promotedPiece;

            m_pieces[move.to] = toPiece;
            m_pieces[move.from] = Piece::none();

            zobrist ^= Zobrist::psq[fromPiece][move.from] ^ Zobrist::psq[toPiece][move.to];

            m_pieceBB[fromPiece] ^= move.from;
            m_pieceBB[toPiece] ^= move.to;

            m_pieceBB[capturedPiece] ^= move.to;
            m_pieceBB[Piece::none()] ^= move.from;

            m_piecesByColorBB[fromPiece.color()] ^= move.to;
            m_piecesByColorBB[fromPiece.color()] ^= move.from;
            if (capturedPiece != Piece::none())
            {
                m_piecesByColorBB[capturedPiece.color()] ^= move.to;
                --m_pieceCount[capturedPiece];
                ++m_pieceCount[Piece::none()];

                zobrist ^= Zobrist::psq[capturedPiece][move.to];
            }

            --m_pieceCount[fromPiece];
            ++m_pieceCount[toPiece];

            return capturedPiece;
        }
        else if (move.type == MoveType::EnPassant)
        {
            const Piece movedPiece = m_pieces[move.from];
            const Piece capturedPiece(PieceType::Pawn, !movedPiece.color());
            const Square capturedPieceSq(move.to.file(), move.from.rank());

            // on ep move there are 3 squares involved
            m_pieces[move.to] = movedPiece;
            m_pieces[move.from] = Piece::none();
            m_pieces[capturedPieceSq] = Piece::none();

            zobrist ^= Zobrist::psq[movedPiece][move.from] ^ Zobrist::psq[movedPiece][move.to];
            zobrist ^= Zobrist::psq[capturedPiece][capturedPieceSq];

            m_pieceBB[movedPiece] ^= move.from;
            m_pieceBB[movedPiece] ^= move.to;

            m_pieceBB[Piece::none()] ^= move.from;
            m_pieceBB[Piece::none()] ^= move.to;

            m_pieceBB[capturedPiece] ^= capturedPieceSq;
            m_pieceBB[Piece::none()] ^= capturedPieceSq;

            m_piecesByColorBB[movedPiece.color()] ^= move.to;
            m_piecesByColorBB[movedPiece.color()] ^= move.from;
            m_piecesByColorBB[capturedPiece.color()] ^= capturedPieceSq;

            --m_pieceCount[capturedPiece];
            ++m_pieceCount[Piece::none()];

            return capturedPiece;
        }
        else // if (move.type == MoveType::Castle)
        {
            const Square rookFromSq = move.to;
            const Square kingFromSq = move.from;

            const Piece rook = m_pieces[rookFromSq];
            const Piece king = m_pieces[kingFromSq];
            const Color color = king.color();

            const CastleType castleType = CastlingTraits::moveCastlingType(move);
            const Square rookToSq = CastlingTraits::rookDestination[color][castleType];
            const Square kingToSq = CastlingTraits::kingDestination[color][castleType];

            // 4 squares are involved
            m_pieces[rookFromSq] = Piece::none();
            m_pieces[kingFromSq] = Piece::none();
            m_pieces[rookToSq] = rook;
            m_pieces[kingToSq] = king;

            zobrist ^= Zobrist::psq[rook][rookFromSq] ^ Zobrist::psq[rook][rookToSq];
            zobrist ^= Zobrist::psq[king][kingFromSq] ^ Zobrist::psq[king][kingToSq];

            m_pieceBB[rook] ^= rookFromSq;
            m_pieceBB[rook] ^= rookToSq;

            m_pieceBB[king] ^= kingFromSq;
            m_pieceBB[king] ^= kingToSq;

            m_pieceBB[Piece::none()] ^= rookFromSq;
            m_pieceBB[Piece::none()] ^= rookToSq;

            m_pieceBB[Piece::none()] ^= kingFromSq;
            m_pieceBB[Piece::none()] ^= kingToSq;

            m_piecesByColorBB[color] ^= rookFromSq;
            m_piecesByColorBB[color] ^= rookToSq;
            m_piecesByColorBB[color] ^= kingFromSq;
            m_piecesByColorBB[color] ^= kingToSq;

            return Piece::none();
        }
    }

    constexpr void undoMove(Move move, Piece capturedPiece)
    {
        if (move.type == MoveType::Normal || move.type == MoveType::Promotion)
        {
            const Piece toPiece = m_pieces[move.to];
            const Piece fromPiece = move.promotedPiece == Piece::none() ? toPiece : Piece(PieceType::Pawn, toPiece.color());

            m_pieces[move.from] = fromPiece;
            m_pieces[move.to] = capturedPiece;

            m_pieceBB[fromPiece] ^= move.from;
            m_pieceBB[toPiece] ^= move.to;

            m_pieceBB[capturedPiece] ^= move.to;
            m_pieceBB[Piece::none()] ^= move.from;

            m_piecesByColorBB[fromPiece.color()] ^= move.to;
            m_piecesByColorBB[fromPiece.color()] ^= move.from;
            if (capturedPiece != Piece::none())
            {
                m_piecesByColorBB[capturedPiece.color()] ^= move.to;
                ++m_pieceCount[capturedPiece];
                --m_pieceCount[Piece::none()];
            }

            if (move.type == MoveType::Promotion)
            {
                --m_pieceCount[toPiece];
                ++m_pieceCount[fromPiece];
            }
        }
        else if (move.type == MoveType::EnPassant)
        {
            const Piece movedPiece = m_pieces[move.to];
            const Piece capturedPiece(PieceType::Pawn, !movedPiece.color());
            const Square capturedPieceSq(move.to.file(), move.from.rank());

            m_pieces[move.to] = Piece::none();
            m_pieces[move.from] = movedPiece;
            m_pieces[capturedPieceSq] = capturedPiece;

            m_pieceBB[movedPiece] ^= move.from;
            m_pieceBB[movedPiece] ^= move.to;

            m_pieceBB[Piece::none()] ^= move.from;
            m_pieceBB[Piece::none()] ^= move.to;

            // on ep move there are 3 squares involved
            m_pieceBB[capturedPiece] ^= capturedPieceSq;
            m_pieceBB[Piece::none()] ^= capturedPieceSq;

            m_piecesByColorBB[movedPiece.color()] ^= move.to;
            m_piecesByColorBB[movedPiece.color()] ^= move.from;
            m_piecesByColorBB[capturedPiece.color()] ^= capturedPieceSq;

            ++m_pieceCount[capturedPiece];
            --m_pieceCount[Piece::none()];
        }
        else // if (move.type == MoveType::Castle)
        {
            const Square rookFromSq = move.to;
            const Square kingFromSq = move.from;

            const Color color = move.to.rank() == rank1 ? Color::White : Color::Black;

            const CastleType castleType = CastlingTraits::moveCastlingType(move);
            const Square rookToSq = CastlingTraits::rookDestination[color][castleType];
            const Square kingToSq = CastlingTraits::kingDestination[color][castleType];

            const Piece rook = m_pieces[rookToSq];
            const Piece king = m_pieces[kingToSq];

            // 4 squares are involved
            m_pieces[rookFromSq] = rook;
            m_pieces[kingFromSq] = king;
            m_pieces[rookToSq] = Piece::none();
            m_pieces[kingToSq] = Piece::none();

            m_pieceBB[rook] ^= rookFromSq;
            m_pieceBB[rook] ^= rookToSq;

            m_pieceBB[king] ^= kingFromSq;
            m_pieceBB[king] ^= kingToSq;

            m_pieceBB[Piece::none()] ^= rookFromSq;
            m_pieceBB[Piece::none()] ^= rookToSq;

            m_pieceBB[Piece::none()] ^= kingFromSq;
            m_pieceBB[Piece::none()] ^= kingToSq;

            m_piecesByColorBB[color] ^= rookFromSq;
            m_piecesByColorBB[color] ^= rookToSq;
            m_piecesByColorBB[color] ^= kingFromSq;
            m_piecesByColorBB[color] ^= kingToSq;
        }
    }

    // Returns whether a given square is attacked by any piece
    // of `attackerColor` side.
    [[nodiscard]] bool isSquareAttacked(Square sq, Color attackerColor) const;

    // Returns whether a given square is attacked by any piece
    // of `attackerColor` side after `move` is made.
    // Move must be pseudo legal.
    [[nodiscard]] bool isSquareAttackedAfterMove(Move move, Square sq, Color attackerColor) const;

    // Move must be pseudo legal.
    // Must not be a king move.
    [[nodiscard]] bool createsDiscoveredAttackOnOwnKing(Move move) const;

    // Returns whether a piece on a given square is attacked
    // by any enemy piece. False if square is empty.
    [[nodiscard]] bool isPieceAttacked(Square sq) const;

    // Returns whether a piece on a given square is attacked
    // by any enemy piece after `move` is made. False if square is empty.
    // Move must be pseudo legal.
    [[nodiscard]] bool isPieceAttackedAfterMove(Move move, Square sq) const;

    // Returns whether the king of the moving side is attacked
    // by any enemy piece after a move is made.
    // Move must be pseudo legal.
    [[nodiscard]] bool isOwnKingAttackedAfterMove(Move move) const;

    // Return a bitboard with all (pseudo legal) attacks by the piece on
    // the given square. Empty if no piece on the square.
    [[nodiscard]] Bitboard attacks(Square sq) const;

    // Returns a bitboard with all squared that have pieces
    // that attack a given square (pseudo legally)
    [[nodiscard]] Bitboard attackers(Square sq, Color attackerColor) const;

    [[nodiscard]] constexpr Piece pieceAt(Square sq) const
    {
        ASSERT(sq.isOk());

        return m_pieces[sq];
    }

    [[nodiscard]] constexpr Bitboard piecesBB(Color c) const
    {
        return m_piecesByColorBB[c];
    }

    [[nodiscard]] INTRIN_CONSTEXPR Square kingSquare(Color c) const
    {
        return piecesBB(Piece(PieceType::King, c)).first();
    }

    [[nodiscard]] constexpr Bitboard piecesBB(Piece pc) const
    {
        return m_pieceBB[pc];
    }

    [[nodiscard]] constexpr Bitboard piecesBB() const
    {
        Bitboard bb{};

        // don't collect from null piece
        return piecesBB(Color::White) | piecesBB(Color::Black);

        return bb;
    }

    [[nodiscard]] constexpr std::uint8_t pieceCount(Piece pt) const
    {
        return m_pieceCount[pt];
    }

    [[nodiscard]] constexpr bool isPromotion(Square from, Square to) const
    {
        ASSERT(from.isOk() && to.isOk());

        return m_pieces[from].type() == PieceType::Pawn && (to.rank() == rank1 || to.rank() == rank8);
    }

    const Piece* piecesRaw() const;

private:
    EnumArray<Square, Piece> m_pieces;
    EnumArray<Piece, Bitboard> m_pieceBB;
    EnumArray<Color, Bitboard> m_piecesByColorBB;
    EnumArray<Piece, uint8_t> m_pieceCount;

    // NOTE: currently we don't track it because it's not
    // required to perform ep if we don't need to check validity
    // Square m_epSquare = Square::none();

    [[nodiscard]] static constexpr bool bbsEqual(const Board& lhs, const Board& rhs) noexcept
    {
        for (Piece pc : values<Piece>())
        {
            if (lhs.m_pieceBB[pc] != rhs.m_pieceBB[pc])
            {
                return false;
            }
        }

        return true;
    }
};

struct Position;

struct MoveLegalityChecker
{
    MoveLegalityChecker(const Position& position);

    [[nodiscard]] bool isPseudoLegalMoveLegal(const Move& move) const;

private:
    const Position* m_position;
    Bitboard m_checkers;
    Bitboard m_ourBlockersForKing;
    Bitboard m_potentialCheckRemovals;
    Square m_ksq;
};

struct CompressedPosition;

struct PositionHash128
{
    std::uint64_t high;
    std::uint64_t low;
};

struct Position : public Board
{
    using BaseType = Board;

    constexpr Position() noexcept :
        Board(),
        m_sideToMove(Color::White),
        m_epSquare(Square::none()),
        m_castlingRights(CastlingRights::All),
        m_rule50Counter(0),
        m_ply(0)
    {
    }

    constexpr Position(const Board& board, Color sideToMove, Square epSquare, CastlingRights castlingRights) :
        Board(board),
        m_sideToMove(sideToMove),
        m_epSquare(epSquare),
        m_castlingRights(castlingRights),
        m_rule50Counter(0),
        m_ply(0)
    {
    }

    void set(std::string_view fen);

    // Returns false if the fen was not valid
    // If the returned value was false the position
    // is in unspecified state.
    [[nodiscard]] bool trySet(std::string_view fen);

    [[nodiscard]] static Position fromFen(std::string_view fen);

    [[nodiscard]] static std::optional<Position> tryFromFen(std::string_view fen);

    [[nodiscard]] static Position startPosition();

    [[nodiscard]] std::string fen() const;

    constexpr void setEpSquareUnchecked(Square sq)
    {
        m_epSquare = sq;
    }

    void setEpSquare(Square sq)
    {
        m_epSquare = sq;
        nullifyEpSquareIfNotPossible();
    }

    constexpr void setSideToMove(Color color)
    {
        m_sideToMove = color;
    }

    constexpr void addCastlingRights(CastlingRights rights)
    {
        m_castlingRights |= rights;
    }

    constexpr void setCastlingRights(CastlingRights rights)
    {
        m_castlingRights = rights;
    }

    constexpr void setRule50Counter(std::uint8_t v)
    {
        m_rule50Counter = v;
    }

    constexpr void setPly(std::uint16_t ply)
    {
        m_ply = ply;
    }

    ReverseMove doMove(const Move& move);

    constexpr void undoMove(const ReverseMove& reverseMove)
    {
        const Move& move = reverseMove.move;
        BaseType::undoMove(move, reverseMove.capturedPiece);

        m_epSquare = reverseMove.oldEpSquare;
        m_castlingRights = reverseMove.oldCastlingRights;

        m_sideToMove = !m_sideToMove;

        --m_ply;
        if (m_rule50Counter > 0)
        {
            m_rule50Counter -= 1;
        }
    }

    [[nodiscard]] constexpr Color sideToMove() const
    {
        return m_sideToMove;
    }

    [[nodiscard]] std::uint8_t rule50Counter() const
    {
        return m_rule50Counter;
    }

    [[nodiscard]] std::uint16_t ply() const
    {
        return m_ply;
    }

    [[nodiscard]] std::uint16_t halfMove() const
    {
        return (m_ply + 1) / 2;
    }

    void setHalfMove(std::uint16_t hm)
    {
        m_ply = 2 * hm - 1 + (m_sideToMove == Color::Black);
    }

    [[nodiscard]] bool isCheck() const;

    [[nodiscard]] Bitboard checkers() const;

    [[nodiscard]] bool isCheckAfterMove(Move move) const;

    // Checks whether ANY `move` is legal.
    [[nodiscard]] bool isMoveLegal(Move move) const;

    [[nodiscard]] bool isPseudoLegalMoveLegal(Move move) const;

    [[nodiscard]] bool isMovePseudoLegal(Move move) const;

    // Returns all pieces that block a slider
    // from attacking our king. When two or more
    // pieces block a single slider then none
    // of these pieces are included.
    [[nodiscard]] Bitboard blockersForKing(Color color) const;

    [[nodiscard]] MoveLegalityChecker moveLegalityChecker() const
    {
        return { *this };
    }

    [[nodiscard]] constexpr Square epSquare() const
    {
        return m_epSquare;
    }

    [[nodiscard]] constexpr CastlingRights castlingRights() const
    {
        return m_castlingRights;
    }

    [[nodiscard]] constexpr bool friend operator==(const Position& lhs, const Position& rhs) noexcept
    {
        return
            lhs.m_sideToMove == rhs.m_sideToMove
            && lhs.m_epSquare == rhs.m_epSquare
            && lhs.m_castlingRights == rhs.m_castlingRights
            && static_cast<const Board&>(lhs) == static_cast<const Board&>(rhs);
    }

    [[nodiscard]] constexpr bool friend operator!=(const Position& lhs, const Position& rhs) noexcept
    {
        return !(lhs == rhs);
    }

    // these are supposed to be used only for testing
    // that's why there's this assert in afterMove

    [[nodiscard]] constexpr Position beforeMove(const ReverseMove& reverseMove) const
    {
        Position cpy(*this);
        cpy.undoMove(reverseMove);
        return cpy;
    }

    [[nodiscard]] Position afterMove(Move move) const;

    [[nodiscard]] constexpr bool isEpPossible() const
    {
        return m_epSquare != Square::none();
    }

    [[nodiscard]] INTRIN_CONSTEXPR CompressedPosition compress() const;

protected:
    Color m_sideToMove;
    Square m_epSquare;
    CastlingRights m_castlingRights;
    std::uint8_t m_rule50Counter;
    std::uint16_t m_ply;

    static_assert(sizeof(Color) + sizeof(Square) + sizeof(CastlingRights) + sizeof(std::uint8_t) == 4);

    [[nodiscard]] FORCEINLINE bool isEpPossible(Square epSquare, Color sideToMove) const;

    [[nodiscard]] NOINLINE bool isEpPossibleColdPath(Square epSquare, Bitboard pawnsAttackingEpSquare, Color sideToMove) const;

    void nullifyEpSquareIfNotPossible();
};

struct PositionWithZobrist : public Position
{
    using BaseType = Board;

    constexpr PositionWithZobrist() noexcept :
        Position(),
        m_zobrist{}
    {
        initZobrist();
    }

    constexpr PositionWithZobrist(const Board& board, Color sideToMove, Square epSquare, CastlingRights castlingRights) :
        Position(board, sideToMove, epSquare, castlingRights),
        m_zobrist{}
    {
        initZobrist();
    }

    explicit constexpr PositionWithZobrist(const Position& pos) :
        Position(pos)
    {
        initZobrist();
    }

    void set(const char* fen);

    // Returns false if the fen was not valid
    // If the returned value was false the position
    // is in unspecified state.
    [[nodiscard]] bool trySet(std::string_view fen);

    [[nodiscard]] static PositionWithZobrist fromFen(const char* fen);

    [[nodiscard]] static std::optional<PositionWithZobrist> tryFromFen(std::string_view fen);

    [[nodiscard]] static PositionWithZobrist startPosition();

    constexpr void setEpSquareUnchecked(Square sq)
    {
        if (m_epSquare != Square::none())
        {
            m_zobrist ^= Zobrist::enpassant[m_epSquare.file()];
        }

        m_epSquare = sq;

        if (m_epSquare != Square::none())
        {
            m_zobrist ^= Zobrist::enpassant[m_epSquare.file()];
        }
    }

    constexpr void setSideToMove(Color color)
    {
        if (m_sideToMove != color)
        {
            m_zobrist ^= Zobrist::blackToMove;
            m_sideToMove = color;
        }
    }

    constexpr void addCastlingRights(CastlingRights rights)
    {
        const auto oldCastlingRights = m_castlingRights;
        m_castlingRights |= rights;

        if (oldCastlingRights != m_castlingRights)
        {
            m_zobrist ^=
                Zobrist::castling[static_cast<unsigned>(oldCastlingRights)]
                ^ Zobrist::castling[static_cast<unsigned>(m_castlingRights)];
        }
    }

    constexpr void setCastlingRights(CastlingRights rights)
    {
        const auto oldCastlingRights = m_castlingRights;
        m_castlingRights = rights;

        if (oldCastlingRights != m_castlingRights)
        {
            m_zobrist ^=
                Zobrist::castling[static_cast<unsigned>(oldCastlingRights)]
                ^ Zobrist::castling[static_cast<unsigned>(m_castlingRights)];
        }
    }

    constexpr void place(Piece piece, Square sq)
    {
        Board::place(piece, sq, m_zobrist);
    }

    ReverseMove doMove(const Move& move);

    constexpr void undoMove(const ReverseMove& reverseMove) = delete;

    [[nodiscard]] constexpr bool friend operator==(const PositionWithZobrist& lhs, const PositionWithZobrist& rhs) noexcept
    {
        return
            lhs.m_zobrist == rhs.m_zobrist
            && lhs.m_sideToMove == rhs.m_sideToMove
            && lhs.m_epSquare == rhs.m_epSquare
            && lhs.m_castlingRights == rhs.m_castlingRights
            && static_cast<const Board&>(lhs) == static_cast<const Board&>(rhs);
    }

    [[nodiscard]] constexpr Position beforeMove(const ReverseMove& reverseMove) const = delete;

    [[nodiscard]] PositionWithZobrist afterMove(Move move) const;

    [[nodiscard]] ZobristKey zobrist() const;

private:
    ZobristKey m_zobrist;

    constexpr void initZobrist()
    {
        m_zobrist = Zobrist::zero;
        m_zobrist ^= Zobrist::castling[static_cast<unsigned>(m_castlingRights)];
        if (m_epSquare != Square::none())
        {
            m_zobrist ^= Zobrist::enpassant[m_epSquare.file()];
        }

        if (m_sideToMove == Color::Black)
        {
            m_zobrist ^= Zobrist::blackToMove;
        }

        for (auto sq : values<Square>())
        {
            const auto piece = pieceAt(sq);
            if (piece != Piece::none())
            {
                m_zobrist ^= Zobrist::psq[piece][sq];
            }
        }
    }
};

struct CompressedPosition
{
    friend struct Position;

    // Occupied bitboard has bits set for
    // each square with a piece on it.
    // Each packedState byte holds 2 values (nibbles).
    // First one at low bits, second one at high bits.
    // Values correspond to consecutive squares
    // in bitboard iteration order.
    // Nibble values:
    // these are the same as for Piece
    // knights, bishops, queens can just be copied
    //  0 : white pawn
    //  1 : black pawn
    //  2 : white knight
    //  3 : black knight
    //  4 : white bishop
    //  5 : black bishop
    //  6 : white rook
    //  7 : black rook
    //  8 : white queen
    //  9 : black queen
    // 10 : white king
    // 11 : black king
    //
    // these are special
    // 12 : pawn with ep square behind (white or black, depending on rank)
    // 13 : white rook with coresponding castling rights
    // 14 : black rook with coresponding castling rights
    // 15 : black king and black is side to move
    //
    // Let N be the number of bits set in occupied bitboard.
    // Only N nibbles are present. (N+1)/2 bytes are initialized.

    static CompressedPosition readFromBigEndian(const unsigned char* data)
    {
        CompressedPosition pos{};
        pos.m_occupied = Bitboard::fromBits(
            (std::uint64_t)data[0] << 56
            | (std::uint64_t)data[1] << 48
            | (std::uint64_t)data[2] << 40
            | (std::uint64_t)data[3] << 32
            | (std::uint64_t)data[4] << 24
            | (std::uint64_t)data[5] << 16
            | (std::uint64_t)data[6] << 8
            | (std::uint64_t)data[7]
            );
        std::memcpy(pos.m_packedState, data + 8, 16);
        return pos;
    }

    constexpr CompressedPosition() :
        m_occupied{},
        m_packedState{}
    {
    }

    [[nodiscard]] friend bool operator<(const CompressedPosition& lhs, const CompressedPosition& rhs)
    {
        if (lhs.m_occupied.bits() < rhs.m_occupied.bits()) return true;
        if (lhs.m_occupied.bits() > rhs.m_occupied.bits()) return false;

        return std::strcmp(reinterpret_cast<const char*>(lhs.m_packedState), reinterpret_cast<const char*>(rhs.m_packedState)) < 0;
    }

    [[nodiscard]] friend bool operator==(const CompressedPosition& lhs, const CompressedPosition& rhs)
    {
        return lhs.m_occupied == rhs.m_occupied
            && std::strcmp(reinterpret_cast<const char*>(lhs.m_packedState), reinterpret_cast<const char*>(rhs.m_packedState)) == 0;
    }

    [[nodiscard]] INTRIN_CONSTEXPR Position decompress() const;

    [[nodiscard]] constexpr Bitboard pieceBB() const
    {
        return m_occupied;
    }

    void writeToBigEndian(unsigned char* data)
    {
        const auto occupied = m_occupied.bits();
        *data++ = occupied >> 56;
        *data++ = (occupied >> 48) & 0xFF;
        *data++ = (occupied >> 40) & 0xFF;
        *data++ = (occupied >> 32) & 0xFF;
        *data++ = (occupied >> 24) & 0xFF;
        *data++ = (occupied >> 16) & 0xFF;
        *data++ = (occupied >> 8) & 0xFF;
        *data++ = occupied & 0xFF;
        std::memcpy(data, m_packedState, 16);
    }

private:
    Bitboard m_occupied;
    std::uint8_t m_packedState[16];
};

static_assert(sizeof(CompressedPosition) == 24);
static_assert(std::is_trivially_copyable_v<CompressedPosition>);

namespace detail
{
    [[nodiscard]] FORCEINLINE constexpr std::uint8_t compressOrdinaryPiece(const Position&, Square, Piece piece)
    {
        return static_cast<std::uint8_t>(ordinal(piece));
    }

    [[nodiscard]] FORCEINLINE constexpr std::uint8_t compressPawn(const Position& position, Square sq, Piece piece)
    {
        const Square epSquare = position.epSquare();
        if (epSquare == Square::none())
        {
            return static_cast<std::uint8_t>(ordinal(piece));
        }
        else
        {
            const Color sideToMove = position.sideToMove();
            const Rank rank = sq.rank();
            const File file = sq.file();
            // use bitwise operators, there is a lot of unpredictable branches but in
            // total the result is quite predictable
            if (
                (file == epSquare.file())
                && (
                ((rank == rank4) & (sideToMove == Color::Black))
                    | ((rank == rank5) & (sideToMove == Color::White))
                    )
                )
            {
                return 12;
            }
            else
            {
                return static_cast<std::uint8_t>(ordinal(piece));
            }
        }
    }

    [[nodiscard]] FORCEINLINE constexpr std::uint8_t compressRook(const Position& position, Square sq, Piece piece)
    {
        const CastlingRights castlingRights = position.castlingRights();
        const Color color = piece.color();

        if (color == Color::White
            && (
            (sq == a1 && contains(castlingRights, CastlingRights::WhiteQueenSide))
                || (sq == h1 && contains(castlingRights, CastlingRights::WhiteKingSide))
                )
            )
        {
            return 13;
        }
        else if (
            color == Color::Black
            && (
            (sq == a8 && contains(castlingRights, CastlingRights::BlackQueenSide))
                || (sq == h8 && contains(castlingRights, CastlingRights::BlackKingSide))
                )
            )
        {
            return 14;
        }
        else
        {
            return static_cast<std::uint8_t>(ordinal(piece));
        }
    }

    [[nodiscard]] FORCEINLINE constexpr std::uint8_t compressKing(const Position& position, Square sq, Piece piece)
    {
        const Color color = piece.color();
        const Color sideToMove = position.sideToMove();

        if (color == Color::White)
        {
            return 10;
        }
        else if (sideToMove == Color::White)
        {
            return 11;
        }
        else
        {
            return 15;
        }
    }
}

namespace detail::lookup
{
    static constexpr EnumArray<PieceType, std::uint8_t(*)(const Position&, Square, Piece)> pieceCompressorFunc = []() {
        EnumArray<PieceType, std::uint8_t(*)(const Position&, Square, Piece)> pieceCompressorFunc{};

        pieceCompressorFunc[PieceType::Knight] = detail::compressOrdinaryPiece;
        pieceCompressorFunc[PieceType::Bishop] = detail::compressOrdinaryPiece;
        pieceCompressorFunc[PieceType::Queen] = detail::compressOrdinaryPiece;

        pieceCompressorFunc[PieceType::Pawn] = detail::compressPawn;
        pieceCompressorFunc[PieceType::Rook] = detail::compressRook;
        pieceCompressorFunc[PieceType::King] = detail::compressKing;

        pieceCompressorFunc[PieceType::None] = [](const Position&, Square, Piece) -> std::uint8_t { /* should never happen */ return 0; };

        return pieceCompressorFunc;
    }();
}

[[nodiscard]] INTRIN_CONSTEXPR CompressedPosition Position::compress() const
{
    auto compressPiece = [this](Square sq, Piece piece) -> std::uint8_t {
        if (piece.type() == PieceType::Pawn) // it's likely to be a pawn
        {
            return detail::compressPawn(*this, sq, piece);
        }
        else
        {
            return detail::lookup::pieceCompressorFunc[piece.type()](*this, sq, piece);
        }
    };

    const Bitboard occ = piecesBB();

    CompressedPosition compressed;
    compressed.m_occupied = occ;

    auto it = occ.begin();
    auto end = occ.end();
    for (int i = 0;; ++i)
    {
        if (it == end) break;
        compressed.m_packedState[i] = compressPiece(*it, pieceAt(*it));
        ++it;

        if (it == end) break;
        compressed.m_packedState[i] |= compressPiece(*it, pieceAt(*it)) << 4;
        ++it;
    }

    return compressed;
}

[[nodiscard]] INTRIN_CONSTEXPR Position CompressedPosition::decompress() const
{
    Position pos;
    pos.setCastlingRights(CastlingRights::None);

    auto decompressPiece = [&pos](Square sq, std::uint8_t nibble) {
        switch (nibble)
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        {
            pos.place(fromOrdinal<Piece>(nibble), sq);
            return;
        }

        case 12:
        {
            const Rank rank = sq.rank();
            if (rank == rank4)
            {
                pos.place(whitePawn, sq);
                pos.setEpSquareUnchecked(sq + Offset{ 0, -1 });
            }
            else // (rank == rank5)
            {
                pos.place(blackPawn, sq);
                pos.setEpSquareUnchecked(sq + Offset{ 0, 1 });
            }
            return;
        }

        case 13:
        {
            pos.place(whiteRook, sq);
            if (sq == a1)
            {
                pos.addCastlingRights(CastlingRights::WhiteQueenSide);
            }
            else // (sq == H1)
            {
                pos.addCastlingRights(CastlingRights::WhiteKingSide);
            }
            return;
        }

        case 14:
        {
            pos.place(blackRook, sq);
            if (sq == a8)
            {
                pos.addCastlingRights(CastlingRights::BlackQueenSide);
            }
            else // (sq == H8)
            {
                pos.addCastlingRights(CastlingRights::BlackKingSide);
            }
            return;
        }

        case 15:
        {
            pos.place(blackKing, sq);
            pos.setSideToMove(Color::Black);
            return;
        }

        }

        return;
    };

    const Bitboard occ = m_occupied;

    auto it = occ.begin();
    auto end = occ.end();
    for (int i = 0;; ++i)
    {
        if (it == end) break;
        decompressPiece(*it, m_packedState[i] & 0xF);
        ++it;

        if (it == end) break;
        decompressPiece(*it, m_packedState[i] >> 4);
        ++it;
    }

    return pos;
}
