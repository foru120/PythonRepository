import queue
# Class for 1 horses in the chess board
class Horse:
    def __init__(self, pos, idx):
        self.pos = pos
        self.idx = idx

    # Return position of this horse
    def get_pos(self):
        return self.pos

    # Return idx of this horse
    def get_idx(self):
        return self.idx

    # Return another horse moved from this horse
    def move_and_create(self, dpos):
        return Horse(self.pos+dpos, self.idx + 1)

# Main
q = queue.Queue()
delta = (-2-1j, -1-2j, 1-2j, 2-1j, 2+1j, 1+2j, -1+2j, -2+1j)

# Input, initial setting
N, M = map(int, input().split(' '))
R, C, S, K = map(int, input().split(' '))
start, end = complex(R-1, C-1), complex(S-1, K-1)
q.put(Horse(start, 0))
visit = {}

finished = False
result = 0

# BFS
while not q.empty() and not finished:
    # Get the top of the queue
    out_horse = q.get()
    in_horse = None

    for i in range(8):
        # Find near position to get from out_horse horse
        in_horse = out_horse.move_and_create(delta[i])
        pos, x, y = in_horse.get_pos(), in_horse.get_pos().real, in_horse.get_pos().imag

        # Horse is out of the map
        if x < 0 or x >= N or y < 0 or y >= M or visit.get(pos) is not None:
            continue

        # When the horse get the end point
        if pos == end:
            result = in_horse.get_idx()
            finished = True
            break

        # Put this horse to the queue
        if not finished:
            q.put(in_horse)
            visit[pos] = 1

# Print result
print(result)