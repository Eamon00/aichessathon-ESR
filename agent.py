"""Chessathon V2: iterative-deepening alpha-beta chess engine."""

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
GAME_TT = {}
GAME_KILLERS = {}
GAME_HISTORY = {}
SEEN_POSITIONS = {}
MAX_TT_ENTRIES = 250_000

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
        remaining = time_left_ms / 1000.0
        budget = min(1.0, remaining / 60.0 + 0.070)
        self.deadline = time.perf_counter() + max(0.015, budget)
        self.nodes = 0

        # Retained for the lifetime of this game process.
        self.tt = GAME_TT
        self.killers = GAME_KILLERS
        self.history = GAME_HISTORY

    def check_time(self):
        self.nodes += 1
        if self.nodes & 31 == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    def evaluate(self, board: chess.Board) -> int:
        """Static evaluation from the side-to-move's perspective."""
        score = 0

        for piece_type, value in PIECE_VALUES.items():
            for square in board.pieces(piece_type, chess.WHITE):
                score += value + PSTS[piece_type][square]

            for square in board.pieces(piece_type, chess.BLACK):
                score -= value + PSTS[piece_type][chess.square_mirror(square)]

        mobility = board.legal_moves.count()
        board.push(chess.Move.null())
        try:
            opponent_mobility = board.legal_moves.count()
        finally:
            board.pop()

        score += 3 * (mobility - opponent_mobility)
        return score if board.turn == chess.WHITE else -score

    def move_order_score(self, board: chess.Board, move: chess.Move) -> int:
        score = 0

        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)

            victim_value = (
                PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
            )
            attacker_value = (
                PIECE_VALUES.get(attacker.piece_type, 20_000) if attacker else 0
            )
            score += 10_000 + 10 * victim_value - attacker_value

        if move.promotion:
            score += 8_000 + PIECE_VALUES.get(move.promotion, 0)

        if board.gives_check(move):
            score += 7_000

        if move in self.killers.get(board.ply(), ()):
            score += 6_000

        return score + self.history.get(move, 0)

    def ordered_moves(
        self, board: chess.Board, tt_move: chess.Move | None = None
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(
            key=lambda move: self.move_order_score(board, move),
            reverse=True,
        )

        if tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)

        return moves

    def quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        self.check_time()

        if board.is_checkmate():
            return -MATE_SCORE + board.ply()
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # In check, every legal evasion must be searched.
        if board.is_check():
            moves = self.ordered_moves(board)
        else:
            stand_pat = self.evaluate(board)

            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat

            moves = [
                move
                for move in board.legal_moves
                if board.is_capture(move) or move.promotion
            ]
            moves.sort(
                key=lambda move: self.move_order_score(board, move),
                reverse=True,
            )

        for move in moves:
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha)
            finally:
                board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int
    ) -> int:
        self.check_time()

        if board.is_checkmate():
            return -MATE_SCORE + board.ply()
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        key = board._transposition_key()
        entry = self.tt.get(key)
        tt_move = entry[3] if entry else None

        original_alpha = alpha
        original_beta = beta

        if entry is not None:
            stored_depth, stored_score, flag, _ = entry

            if stored_depth >= depth:
                if flag == 0:
                    return stored_score
                if flag == -1:
                    beta = min(beta, stored_score)
                elif flag == 1:
                    alpha = max(alpha, stored_score)

                if alpha >= beta:
                    return stored_score

        best_score = -INF
        best_move = None

        for move in self.ordered_moves(board, tt_move):
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

            if alpha >= beta:
                killers = self.killers.setdefault(board.ply(), [])
                if move not in killers:
                    killers.insert(0, move)
                    del killers[2:]

                self.history[move] = self.history.get(move, 0) + depth * depth
                break

        if best_score <= original_alpha:
            flag = -1  # Upper bound.
        elif best_score >= original_beta:
            flag = 1  # Lower bound.
        else:
            flag = 0  # Exact score.

        self.tt[key] = (depth, best_score, flag, best_move)
        return best_score

    def search(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        best_move = legal_moves[0]

        for depth in range(1, 64):
            try:
                alpha = -INF
                beta = INF
                iteration_best = best_move
                iteration_score = -INF

                root_entry = self.tt.get(board._transposition_key())
                root_tt_move = root_entry[3] if root_entry else None

                for move in self.ordered_moves(board, root_tt_move):
                    self.check_time()

                    board.push(move)
                    try:
                        score = -self.negamax(board, depth - 1, -beta, -alpha)

                        # Prefer a similarly scored move that does not complete a
                        # threefold repetition. Do not interfere with mating lines.
                        if (
                            SEEN_POSITIONS.get(board._transposition_key(), 0) >= 2
                            and score < MATE_SCORE - 1_000
                        ):
                            score -= 75
                    finally:
                        board.pop()

                    if score > iteration_score:
                        iteration_score = score
                        iteration_best = move

                    alpha = max(alpha, score)

                best_move = iteration_best

                if iteration_score >= MATE_SCORE - 100:
                    break

            except SearchTimeout:
                break

        return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)

    current_key = board._transposition_key()
    SEEN_POSITIONS[current_key] = SEEN_POSITIONS.get(current_key, 0) + 1

    # Bound memory while retaining useful search information during a game.
    if len(GAME_TT) > MAX_TT_ENTRIES:
        GAME_TT.clear()

    move = Engine(time_left_ms).search(board)

    if move is None:
        return "0000"

    # Remember our chosen resulting position for later threefold avoidance.
    board.push(move)
    next_key = board._transposition_key()
    SEEN_POSITIONS[next_key] = SEEN_POSITIONS.get(next_key, 0) + 1

    return move.uci()