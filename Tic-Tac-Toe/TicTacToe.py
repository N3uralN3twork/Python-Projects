def main():
    board = create_board()
    game_over = False
    player = True # True = Player 1
    while not game_over:
        print_board(board)
        selection = get_selection()
        move = make_move(selection, player, board)
        if move == False:
            print("Invalid selection. Try again.")
            continue
        if is_winning_move(board):
            game_over = True
            break
        player = not player
    winner = 1 if player else 2
    print_board(board)
    print(f"Player {winner} is the winner!")


def check_horizontal(board):
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != "_":
            return True
    return False


def check_diagonal(board):
    if board[1][1] != "_":
        if board[0][0] == board[1][1] == board[2][2]:
            return True
        if board[0][2] == board[1][1] == board[2][0]:
            return True
    return False


def check_vertical(board):
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != "_":
            return True
    return False


def create_board():
    return [["_" for _ in range(3)] for _ in range(3)]


def is_winning_move(board):
    return check_horizontal(board) or check_vertical(board) \
            or check_diagonal(board)


def get_selection():
    try:
        selection = int(input("Choose a square: "))
        if selection < 0 or selection > 8:
            raise ValueError
        return selection
    except ValueError:
        return None


def make_move(selection, player, board):
    if selection is None:
        return False
    row, col = selection // 3, selection % 3
    if board[row][col] == "_":
        board[row][col] = 'X' if player else 'O'
        return True
    else:
        return False


def print_board(board):
    for i in range(len(board)):
        print(f"{i*3}, {i*3+1}, {i*3+2}\t", board[i])


if __name__ == "__main__":
    main()