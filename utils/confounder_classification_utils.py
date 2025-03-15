def create_label_from_output(output, output_for_true):
    output = output.lower().replace(" ", "")
    return 1 if output in output_for_true else 0
