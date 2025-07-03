def mcts_search(board: chess.Board, model: ChessNet, simulations: int = 800) -> chess.Move:
    root = MCTSNode(board.copy())
    game_history = [get_game_state_key(root.board)]
    
    for _ in range(simulations):
        node = root
        current_history = game_history.copy()
        
        while node.children:
            node = max(node.children, key=lambda x: ucb_score(x))
            current_history.append(get_game_state_key(node.board))
            
            if is_threefold_repetition(current_history) or node.board.halfmove_clock >= 50:
                break
        
        if not node.board.is_game_over() and not is_threefold_repetition(current_history):
            board_tensor = torch.FloatTensor(board_to_tensor(node.board)).unsqueeze(0).to(Config.DEVICE)
            policy_logits, _ = model(board_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()
            
            for move in node.board.legal_moves:
                child_board = node.board.copy(stack=False) 
                child_board.push(move)
                child = MCTSNode(child_board, node)
                child.policy = policy[move_to_index(move)]
                node.children.append(child)

        value = evaluate_position(node.board, model)
        
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    if not root.children:
        return random.choice(list(root.board.legal_moves)) if root.board.legal_moves else None
    
    best_child = max(root.children, key=lambda x: x.visits)
    return best_child.board.peek() if best_child.board.move_stack else random.choice(list(root.board.legal_moves))



"""
def generate_selfplay_game(model, sims_per_move):
    board = chess.Board()
    game_history = []
    state_history = [get_game_state_key(board)]  # Track states
    
    while not board.is_game_over():
        # Terminate early if draw conditions met
        board = enforce_single_king(board)
        if board.is_fifty_moves() or board.is_repetition(3):
            break
            
        move_probs = run_mcts(model, board, sims_per_move, state_history)
        move = sample_move_from_probs(board, move_probs)
        
        game_history.append((board.fen(), move))
        board.push(move)
        board = enforce_single_king(board)
        state_history.append(get_game_state_key(board))
    
    result = 0.0  # Default draw
    if board.is_checkmate():
        result = 1.0 if board.turn == chess.BLACK else -1.0
    
    return [(fen, move, result) for fen, move in game_history]

"""
"""
def generate_selfplay_game(model, sims_per_move):
    board = chess.Board()
    game_history = []
    state_history = {get_game_state_key(board): 1}  # Track state counts for repetition
    
    while not board.is_game_over():
        # Validate and correct board state before each move
        board = enforce_single_king(board)
        # Check termination conditions
        if (board.is_fifty_moves() or 
            board.is_repetition(3) or 
            state_history.get(get_game_state_key(board), 0) >= 3):
            break    
        # Get move from MCTS (pass current state history)
        move_probs = run_mcts(model, board, sims_per_move, state_history)
        move = sample_move_from_probs(board, move_probs)       
        if move not in board.legal_moves:
            print(f"Illegal move generated: {move.uci()}")
            break      
        # Record pre-move state
        game_history.append((board.fen(), move.uci()))  # Store UCI string
        # Apply move and update state
        board.push(move)
        board = enforce_single_king(board)
        # Update state history
        current_key = get_game_state_key(board)
        state_history[current_key] = state_history.get(current_key, 0) + 1
    result = 0.0  # Draw
    if board.is_checkmate():
        result = 1.0 if board.turn == chess.BLACK else -1.0  # Winner is opposite of current turn
    
    return [(fen, move, result) for fen, move in game_history]
"""



"""        
def train_phase2():
    model = ChessNet().to(Config.DEVICE)
    model.load_state_dict(torch.load("/Users/sujithsaisripadam/Desktop/Cool_project/chessai/models/phase1_epoch0.pth"))
    
    for game_idx in range(Config.PHASE2["selfplay_games"]):
        print(f"Simulating game: {game_idx}")
        board = chess.Board()
        game_history = []
        
        while not board.is_game_over():
            move = mcts_search(board, model, Config.PHASE2["sims_per_move"])
            game_history.append((board.fen(), move))  # Store FEN and Move object
            board.push(move)
        
        # Save game with serialized moves
        save_selfplay_game(game_history, Config.PHASE2["output_dir"])
"""