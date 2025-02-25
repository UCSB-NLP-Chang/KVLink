def pad_2d_list(input_list, m, n, padding_value=-1):
    # Adjust each row to have n elements
    padded_list = []
    for row in input_list:
        # Truncate the row or pad it with the padding value
        new_row = row[:n] + [padding_value] * (n - len(row))
        # Ensure the new row has exactly n elements
        new_row = new_row[:n]
        padded_list.append(new_row)

    # Adjust the number of rows to be m
    current_row_count = len(padded_list)
    if current_row_count < m:
        # Add new rows filled with the padding value
        for _ in range(m - current_row_count):
            padded_list.append([padding_value] * n)
    elif current_row_count > m:
        # Truncate the list to have m rows
        padded_list = padded_list[:m]

    return padded_list

def transform_2d_list(input_list, target_value=-1):
    # Use nested list comprehension to create a copy with transformed values
    transformed_list = [
        [0 if item == target_value else 1 for item in row]
        for row in input_list
    ]
    return transformed_list