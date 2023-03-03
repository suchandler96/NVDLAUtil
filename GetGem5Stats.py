import os
import argparse
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(tuple_of_root_dirs_files):
    text = tuple_of_root_dirs_files[0].replace("MB", "000000").replace("kB", "000").replace("KB", "000")
    return [atoi(c) for c in re.split('(\d+)', text)]


def get_data_from_file(output_str, file_name):
    with open(file_name, "r") as fp:
        file_lines = fp.readlines()

    # get nvdla_cycles
    to_output_numbers = ['/', '/', '/', '/', '/', '/', '/', '/', '/']

    cycle_counter = 0
    pr_cache_counter = 0

    for file_line in file_lines:
        file_line_words = file_line.split()
        if len(file_line_words) < 1:
            continue
        if "nvdla_cycles" in file_line_words[0]:
            to_output_numbers[cycle_counter] = file_line_words[1]
            cycle_counter += 1
        elif "pr_cache.overallMissRate::total" in file_line_words[0]:
            to_output_numbers[4 + pr_cache_counter] = file_line_words[1]
            pr_cache_counter += 1
        elif "sh_cache.overallMissRate::total" in file_line_words[0]:
            to_output_numbers[8] = file_line_words[1]

    output_str += ','.join(to_output_numbers)
    output_str += '\n'
    return output_str


def parse_args():
    parser = argparse.ArgumentParser(description="GetGem5Stats.py options")
    parser.add_argument("--get-root-dir", "-d", type=str, required=True,
                        help="path to the root dir containing a set of experiments")
    parser.add_argument("--out-dir", "-o", type=str, default=".",
                        help="path to the directory to store the summary of a set of experiments")

    return parser.parse_args()


def main():
    options = parse_args()
    output_str = "MR=Miss Rate\n"\
                 "name[0],name[1],nvdla[0] cycle,nvdla[1] cycle,nvdla[2] cycle,nvdla[3] cycle,"\
                 "prcache[0] MR,prcache[1] MR,prcache[2] MR,prcache[3] MR,shcache MR\n"
    for root, dirs, files in sorted(os.walk(options.get_root_dir), key=natural_keys):
        if "stats.txt" in files:
            # this is a directory that holds all data for an experiment
            dir_level = root.split(os.sep)
            output_str += (dir_level[-2] + "," + dir_level[-1] + ",")
            output_str = get_data_from_file(output_str, os.path.join(root, "stats.txt"))

    # write output_str to csv file
    with open(os.path.join(options.out_dir, "summary.csv"), "w") as fp:
        fp.write(output_str)


if __name__ == "__main__":
    main()
