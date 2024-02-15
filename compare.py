import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("array_1")
    parser.add_argument("array_2")

    args = parser.parse_args()

    with open(args.array_1, "rb") as f_1:
        array_1 = np.load(f_1)
    with open(args.array_2, "rb") as f_2:
        array_2 = np.load(f_2)

    max_diff = np.max(np.abs(array_1 - array_2))

    print(f"Max difference: {max_diff}")


if __name__ == "__main__":
    main()
