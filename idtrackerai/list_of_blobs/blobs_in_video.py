"""
Compact storage for the blobs of a single chunk.

idtracker.ai indexes ``list_of_blobs.blobs_in_video`` by absolute frame number,
so a chunk of a long imgstore used to be padded with millions of empty lists
before and after the chunk. :class:`BlobsInVideo` behaves like that padded list
(same length, same indexing, same iteration) but only stores the frames of the
chunk, which makes iterating, (un)pickling and saving it much cheaper.
"""

import itertools


class FrozenEmptyFrame(list):
    """Empty blobs_in_frame returned for frames outside of the stored chunk.

    It is shared by all those frames, so it refuses to be mutated (adding a blob
    to one such frame would otherwise add it to all of them). Use
    :meth:`BlobsInVideo.ensure_frame` to get a mutable frame instead.
    """

    def _read_only(self, *args, **kwargs):
        raise TypeError(
            "This frame is outside of the chunk stored in blobs_in_video and cannot be modified. "
            "Call blobs_in_video.ensure_frame(frame_number) first"
        )

    append = extend = insert = __setitem__ = __delitem__ = __iadd__ = __imul__ = _read_only

    def __reduce__(self):
        return (list, ())


EMPTY_FRAME = FrozenEmptyFrame()


class BlobsInVideo:
    """List-like collection of blobs_in_frame indexed by absolute frame number.

    Arguments:
        frames (list): blobs_in_frame of each frame of the chunk
        offset (int): frame number of frames[0]
        total (int): number of frames of the whole video (len of this object)
    """

    def __init__(self, frames, offset=0, total=None):
        self._frames = list(frames)
        self._offset = int(offset)
        if total is None:
            total = self._offset + len(self._frames)
        self._total = int(total)
        assert self._offset >= 0
        assert self._total >= self._offset + len(self._frames)

    @property
    def start(self):
        """First frame number stored in this object"""
        return self._offset

    @property
    def end(self):
        """Last frame number stored in this object + 1"""
        return self._offset + len(self._frames)

    def __len__(self):
        return self._total

    def _normalize_index(self, index):
        if index < 0:
            index += self._total
        if not 0 <= index < self._total:
            raise IndexError(f"frame {index} out of range for blobs_in_video of length {self._total}")
        return index

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._total)
            if step == 1:
                if stop <= start:
                    return []
                before = max(0, min(stop, self.start) - start)
                inside = self._frames[max(start, self.start) - self._offset:max(min(stop, self.end), self.start) - self._offset]
                after = max(0, stop - max(start, self.end))
                return [EMPTY_FRAME] * before + inside + [EMPTY_FRAME] * after
            return [self[i] for i in range(start, stop, step)]

        index = self._normalize_index(index)
        if self.start <= index < self.end:
            return self._frames[index - self._offset]
        return EMPTY_FRAME

    def ensure_frame(self, index):
        """Make sure the frame is stored (and therefore mutable) and return it"""
        index = self._normalize_index(index)
        if index < self.start:
            self._frames = [[] for _ in range(self.start - index)] + self._frames
            self._offset = index
        elif index >= self.end:
            self._frames += [[] for _ in range(index - self.end + 1)]
        return self._frames[index - self._offset]

    def __setitem__(self, index, blobs_in_frame):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._total)
            frame_numbers = range(start, stop, step)
            blobs_in_frames = list(blobs_in_frame)
            if len(frame_numbers) != len(blobs_in_frames):
                raise ValueError("BlobsInVideo does not support resizing slice assignment")
            for frame_number, value in zip(frame_numbers, blobs_in_frames):
                self[frame_number] = value
            return

        index = self._normalize_index(index)
        self.ensure_frame(index)
        self._frames[index - self._offset] = blobs_in_frame

    def __iter__(self):
        return itertools.chain(
            itertools.repeat(EMPTY_FRAME, self.start),
            self._frames,
            itertools.repeat(EMPTY_FRAME, self._total - self.end),
        )

    def __reversed__(self):
        return itertools.chain(
            itertools.repeat(EMPTY_FRAME, self._total - self.end),
            reversed(self._frames),
            itertools.repeat(EMPTY_FRAME, self.start),
        )

    def __add__(self, other):
        return list(self) + list(other)

    def __repr__(self):
        return f"BlobsInVideo(frames {self.start}-{self.end} of {self._total})"


def frame_range(blobs_in_video):
    """Range of frame numbers that may contain blobs.

    For a :class:`BlobsInVideo`, this is the range of stored frames;
    for a plain list, it is the whole list
    """
    if isinstance(blobs_in_video, BlobsInVideo):
        return range(blobs_in_video.start, blobs_in_video.end)
    return range(len(blobs_in_video))
