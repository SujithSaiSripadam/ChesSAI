import chess

import chess

"""
*****************
Trying to test fns here before integrating into the main codebase.
***************** 
"""

def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess.Move to a unique index (0–4671).
    - Normal moves: from_square * 64 + to_square → 0–4095
    - Promotion moves: 4096–4671
    """
    if not move.promotion:
        return move.from_square * 64 + move.to_square

    #Promotion move encoding
    promo_offset = {
        chess.QUEEN: 0,
        chess.ROOK: 1,
        chess.BISHOP: 2,
        chess.KNIGHT: 3
    }

    from_sq = move.from_square
    file = chess.square_file(move.to_square)

    from_sq = move.from_square
    to_file = chess.square_file(move.to_square)  # 0-7 (a-h)
    promo_type = promo_offset[move.promotion]

    #Promotion index = 4096 + (from_square % 8) * 32 + (promo_type * 8) + to_file
    return 4096 + (from_sq % 8) * 32 + promo_type * 8 + to_file


def test_move_to_index():
    assert move_to_index(chess.Move(chess.A1, chess.A2)) == 0 * 64 + 8

    #White promotions (e7->e8=Q, e7->e8=N)
    assert move_to_index(chess.Move(chess.E7, chess.E8, promotion=chess.QUEEN)) == 4096 + 4*32 + 0*8 + 4  # = 4228
    assert move_to_index(chess.Move(chess.E7, chess.E8, promotion=chess.KNIGHT)) == 4096 + 4*32 + 3*8 + 4  # = 4252

    #Black promotions (e2->e1=Q, e2->e1=N)
    assert move_to_index(chess.Move(chess.E2, chess.E1, promotion=chess.QUEEN)) == 4096 + 4*32 + 0*8 + 4  # = 4228 
    assert move_to_index(chess.Move(chess.E2, chess.E1, promotion=chess.KNIGHT)) == 4096 + 4*32 + 3*8 + 4  # = 4252

    #Edge case: Underpromotion (a7->a8=B)
    assert move_to_index(chess.Move(chess.A7, chess.A8, promotion=chess.BISHOP)) == 4096 + 0*32 + 2*8 + 0  # = 4112
    print(f"assrt {move_to_index(chess.Move(chess.A7, chess.A8, promotion=chess.BISHOP))} == ? 4112")
    #White promotion: e7 → e8 = Q
    promo_move = chess.Move(chess.E7, chess.E8, promotion=chess.QUEEN)
    print(f"promo move : {promo_move}, index: {move_to_index(promo_move)}")
    assert 4096 <= move_to_index(promo_move) < 4672

    #Black promotion: e2 → e1 = N
    promo_move_black = chess.Move(chess.E2, chess.E1, promotion=chess.KNIGHT)
    print(f"black promo move : {promo_move_black}, index: {move_to_index(promo_move_black)}")
    assert 4096 <= move_to_index(promo_move_black) < 4672

    print("All tests passed!")



    
if __name__ == "__main__":
    test_move_to_index()
    move = chess.Move(chess.A1, chess.A2)
    print(f"Index for move {move}: {move_to_index(move)}")
    
    promo_move = chess.Move(chess.E2, chess.E1, promotion=chess.QUEEN)
    print(f"Index for promotion move {promo_move}: {move_to_index(promo_move)}")
    
    promo_move_black = chess.Move(chess.E7, chess.E8, promotion=chess.KNIGHT)
    print(f"Index for black promotion move {promo_move_black}: {move_to_index(promo_move_black)}")