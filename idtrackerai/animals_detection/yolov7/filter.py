def keep_best_detections(detections, number_of_animals):

    return sorted(detections, key=lambda detection: detection.conf)[::-1][:number_of_animals]