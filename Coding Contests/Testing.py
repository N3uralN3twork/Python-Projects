import pandas as pd

def schema(Sites, NSubjects):
    matt = []
    for i in range(Sites):
        for j in range(NSubjects):
            matt.append([i, j])
    matt = pd.DataFrame(matt)
    return matt


schema(2, 2)

