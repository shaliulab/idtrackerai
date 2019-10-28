def compute_output_width(width, kernel, padding, stride):
    output_width = int(((width - kernel + 2 * padding) / stride) + 1)
    return output_width
