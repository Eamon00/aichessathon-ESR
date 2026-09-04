"""The submission entrypoint. The platform imports this file and calls get_move."""

"""Chessathon V1: iterative-deepening alpha-beta chess engine."""

import math
import time

import chess


# Centipawn values.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE_SCORE = 1_000_000
INF = 10_000_000

# Piece-square tables from White's perspective.
# Index 0 = a1, index 63 = h8.
PAWN_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
     0,   0,   0,   5,   5,   0,   0,   0,
     5,  10,  10,  10,  10,  10,  10,   5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     0,   0,   0,   0,   0,   0,   0,   0,
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

PSTS = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}


class SearchTimeout(Exception):
    pass


class Engine:
    def __init__(self, time_left_ms: int):
        # Keep a safety margin so we don't lose on time.
        remaining = time_left_ms / 1000.0
        budget = min(1.0, remaining / 60.0 + 0.070)
        self.deadline = time.perf_counter() + max(0.015, budget)

        self.nodes = 0
        self.tt = {}
        self.killers = {}
        self.history = {}

    def check_time(self):
        self.nodes += 1

        # Don't call perf_counter on every node.
        if self.nodes & 31 == 0:
            if time.perf_counter() >= self.deadline:
                raise SearchTimeout

    def evaluate(self, board: chess.Board) -> int:
        """Evaluate from the perspective of the side to move."""
        score = 0

        for piece_type, value in PIECE_VALUES.items():
            for square in board.pieces(piece_type, chess.WHITE):
                score += value + PSTS[piece_type][square]

            for square in board.pieces(piece_type, chess.BLACK):
                mirrored = chess.square_mirror(square)
                score -= value + PSTS[piece_type][mirrored]

        # Small mobility bonus.
        mobility = board.legal_moves.count()

        board.push(chess.Move.null())
        opponent_mobility = board.legal_moves.count()
        board.pop()

        score += 3 * (mobility - opponent_mobility)

        # Convert to side-to-move perspective.
        return score if board.turn == chess.WHITE else -score

    def move_order_score(self, board: chess.Board, move: chess.Move) -> int:
        score = 0

        # Captures first, using MVV-LVA.
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)

            if victim:
                score += 10_000 + 10 * PIECE_VALUES[victim.piece_type]

            attacker = board.piece_at(move.from_square)
            if attacker:
                score -= PIECE_VALUES.get(attacker.piece_type, 20_000)

        # Promotions.
        if move.promotion:
            score += 8_000 + PIECE_VALUES.get(move.promotion, 0)

        # Checks.
        board.push(move)
        if board.is_check():
            score += 7_000
        board.pop()

        # Killer moves.
        killers = self.killers.get(board.ply(), ())
        if move in killers:
            score += 6_000

        # History heuristic.
        score += self.history.get(move, 0)

        return score

    def ordered_moves(self, board: chess.Board):
        moves = list(board.legal_moves)
        moves.sort(
            key=lambda m: self.move_order_score(board, m),
            reverse=True,
        )
        return moves

    def quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        self.check_time()

        stand_pat = self.evaluate(board)

        if stand_pat >= beta:
            return beta

        if stand_pat > alpha:
            alpha = stand_pat

        # Only search tactical moves.
        captures = [
            move for move in board.legal_moves
            if board.is_capture(move) or move.promotion
        ]

        captures.sort(
            key=lambda m: self.move_order_score(board, m),
            reverse=True,
        )

        for move in captures:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta

            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int):
        self.check_time()

        if board.is_checkmate():
            return -MATE_SCORE + board.ply()

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        key = board._transposition_key()

        # TT entries:
        # (depth, score, flag, best_move)
        entry = self.tt.get(key)

        if entry is not None:
            stored_depth, stored_score, flag, best_move = entry

            if stored_depth >= depth:
                if flag == 0:
                    return stored_score
                elif flag == -1 and stored_score <= alpha:
                    return stored_score
                elif flag == 1 and stored_score >= beta:
                    return stored_score

        original_alpha = alpha
        best_move = None
        best_score = -INF

        moves = self.ordered_moves(board)

        for move in moves:
            board.push(move)

            score = -self.negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
            )

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Killer heuristic.
                ply = board.ply()
                killers = self.killers.setdefault(ply, [])

                if move not in killers:
                    killers.insert(0, move)
                    if len(killers) > 2:
                        killers.pop()

                # History heuristic.
                self.history[move] = self.history.get(move, 0) + depth * depth

                break

        # Store TT entry.
        if best_score <= original_alpha:
            flag = -1
        elif best_score >= beta:
            flag = 1
        else:
            flag = 0

        self.tt[key] = (depth, best_score, flag, best_move)

        return best_score

    def search(self, board: chess.Board):
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        # Always have a legal fallback.
        best_move = legal_moves[0]

        # Iterative deepening.
        for depth in range(1, 64):
            try:
                alpha = -INF
                beta = INF
                iteration_best = None
                iteration_score = -INF

                moves = self.ordered_moves(board)

                for move in moves:
                    self.check_time()

                    board.push(move)

                    score = -self.negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                    )

                    board.pop()

                    if score > iteration_score:
                        iteration_score = score
                        iteration_best = move

                    if score > alpha:
                        alpha = score

                if iteration_best is not None:
                    best_move = iteration_best

                # If we've found mate, don't waste time.
                if iteration_score >= MATE_SCORE - 100:
                    break

            except SearchTimeout:
                break

        return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)

    engine = Engine(time_left_ms)
    move = engine.search(board)

    return move.uci()
