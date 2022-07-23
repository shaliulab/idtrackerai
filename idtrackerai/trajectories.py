import numpy as np

class Trajectories:

    def __init__(self, trajectories, start, end, total_length, chunked=False):

        if not chunked:
            assert total_length == trajectories["trajectories"].shape[0]

        self._trajectories = trajectories
        self._start = start
        self._end = end
        self._total_length = total_length

        self._chunked=chunked

    def save(self, trajectories_file):
        if self._start is not None and self._end is not None:

            data=self._trajectories.copy()
            for field in ["areas", "trajectories", "id_probabilities"]:
                data[field]=self._trajectories[field][self._start:self._end].copy()

            tr=self.__class__(
                data,
                self._start,
                self._end,
                self._total_length,
                chunked=True
            )
        else:
            tr=self

        np.save(trajectories_file, tr)


    @classmethod
    def load(cls, trajectories_file):

        tr=np.load(trajectories_file, allow_pickle=True).item()
        if tr._chunked:
            data = tr._trajectories.copy()
            number_of_animals = tr._trajectories["trajectories"].shape[1]
            for field in ["areas", "trajectories", "id_probabilities"]:

                if len(data[field].shape) == 3:
                    unit = [[np.nan, ] * data[field].shape[2],]
                else:
                    unit = [np.nan, ]
                prepend = np.concatenate([np.array([unit * number_of_animals]) for _ in range(tr._start)], axis=0)
                append = np.concatenate([np.array([unit* number_of_animals]) for _ in range(tr._end, tr._total_length)], axis=0)
                data[field] = np.vstack([prepend, data[field], append])

            tr=cls(
                data,
                tr._start,
                tr._end,
                tr._total_length,
                chunked=False
            )
        
        return tr._trajectories

            
def save_trajectories(trajectories_file, trajectories, start=None, end=None):  
    tr=Trajectories(trajectories, start, end, trajectories["trajectories"].shape[0], chunked=False)
    tr.save(trajectories_file)

def load_trajectories(trajectories_file):
    return Trajectories.load(trajectories_file)
