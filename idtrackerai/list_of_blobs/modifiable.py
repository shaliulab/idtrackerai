import copy

class Modifications:
    
    @property
    def original_segmentation(self):
        self._original_segmentation = getattr(self, "_original_segmentation", {})
        return self._original_segmentation


    @property
    def modified_segmentation(self):
        self._modified_segmentation = getattr(self, "_modified_segmentation", {})
        return self._modified_segmentation
    
    def apply_modification(self, frame_number, blobs_in_frame):
        
        for blob in blobs_in_frame:
            blob.next, blob.previous = [], []
            blob._bridge_start, blob._bridge_end = None, None            


        if all([not blob.modified for blob in self.blobs_in_video[frame_number]]):
            # the data stored for this frame has not beed edited by idtrackerai (i.e. it's original)
            original_blobs_in_frame = []
            for blob in self.blobs_in_video[frame_number]:
                blob.next, blob.previous = [], []
                blob._bridge_start, blob._bridge_end = None, None
                original_blob = copy.deepcopy(blob)
                original_blobs_in_frame.append(original_blob) 

            self.original_segmentation[frame_number] = original_blobs_in_frame
            
        self.blobs_in_video[frame_number] = blobs_in_frame
        self.modified_segmentation[frame_number] = blobs_in_frame

    def toggle_modifications(self, value=True):
        
        if value:
            for frame_number in self.modified_segmentation:
                self.blobs_in_video[frame_number] = self.modified_segmentation[frame_number]
                
        else:
            for frame_number in self.original_segmentation:
                self.blobs_in_video[frame_number] = self.original_segmentation[frame_number]