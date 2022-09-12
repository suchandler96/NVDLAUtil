import re
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pylab

pivot_addr_reg_book = ["0x5034", "0xb048", "0xc01c"]
addr_reg_book = [0x5034, 0x503c, 0x507c, 0xa02c, 0xa044, 0xb048, 0xc01c, 0xd070]
corr_map_table = {0x5000: [0x5000, 0x6000], 0xb000: [0xa000, 0xb000], 0xc000: [0xc000, 0xd000]}

input_addr_regs = [0x5034, 0x507c, 0xa02c, 0xa044, 0xc01c]
# omitting 0x503c because we assume it will always be the same with 0x5034

output_addr_regs = [0xb048, 0xd070]
reg_to_neglect = [0x500c, 0x5010, 0x6008, 0xa008, 0xb038]


class TxnSegment:
    def __init__(self, core_reg, core_reg_val, core_line_num, file_name):
        # use numbers instead of strings
        # line num starts from 0
        self.core_reg = core_reg
        self.core_reg_val = core_reg_val
        self.addr_reg_mapping = {core_reg: core_reg_val}
        self.other_reg_mapping = {}
        self.core_line_num = core_line_num
        self.file_name = file_name
        self.matched_target = None  # the pointer to its matched txn segment in the unpartitioned txn file


class AddrUsageDesc:
    def __init__(self, addr):
        self.addr = addr
        self.producers = []
        self.consumers = []
        # pointers to txn segments that are using this address

    def add_consumer(self, consumer):
        self.consumers.append(consumer)

    def add_producer(self, producer):
        self.producers.append(producer)

    def is_internal_data_addr(self):
        return len(self.producers) > 0 and len(self.consumers) > 0


def expand_line_range(txn_words, pivot_line_idx, txn_seg):
    patience_limit = 10
    patience = 10
    pivot_reg_base = int(txn_words[pivot_line_idx][3], 16) & 0xff000
    lower_line_idx = pivot_line_idx
    upper_line_idx = pivot_line_idx

    while True:
        lower_line_idx -= 1

        if lower_line_idx < 0:      # to prevent segfault
            break

        reg_addr = int(txn_words[lower_line_idx][-1], 16)

        if txn_words[lower_line_idx][0] != "write_reg" or reg_addr & 0xff000 not in corr_map_table[pivot_reg_base] or (
                (reg_addr & 0xfff == 0x004) or (reg_addr & 0xfff == 0x000)) or reg_addr in reg_to_neglect:
            patience -= 1
            if patience <= 0:
                break
            else:
                continue

        if reg_addr in addr_reg_book:
            # earlier configuration should be overwritten by a later one
            if reg_addr not in txn_seg.addr_reg_mapping:
                txn_seg.addr_reg_mapping[reg_addr] = int(txn_words[lower_line_idx][2], 16)
        else:
            if reg_addr not in txn_seg.other_reg_mapping:
                txn_seg.other_reg_mapping[reg_addr] = int(txn_words[lower_line_idx][2], 16)

    patience = patience_limit

    while True:
        upper_line_idx += 1

        if upper_line_idx >= len(txn_words):    # to prevent segfault
            break

        reg_addr = int(txn_words[upper_line_idx][-1], 16)

        if txn_words[upper_line_idx][0] != "write_reg" or reg_addr & 0xff000 not in corr_map_table[pivot_reg_base] or (
                (reg_addr & 0xfff == 0x004) or (reg_addr & 0xfff == 0x000)) or reg_addr in reg_to_neglect:
            patience -= 1
            if patience <= 0:
                break
            else:
                continue

        if reg_addr in addr_reg_book:
            txn_seg.addr_reg_mapping[reg_addr] = int(txn_words[upper_line_idx][2], 16)
        else:
            txn_seg.other_reg_mapping[reg_addr] = int(txn_words[upper_line_idx][2], 16)


# assume all 0x5034 configs will appear right after a 0x6xxx config
# assume all 0xc01c and 0xd070 configs will come together
# assume all 0xaxxx and 0xbxxx configs will come together
def get_txn_segments(txn_words, file_name):
    line_num = len(txn_words)

    txn_segments = []

    for i in range(line_num):
        if len(txn_words[i]) < 4:
            continue

        # words[0]: write/read_reg, words[1]: 0xffffxxxx, words[2]: value, words[3]: #+reg
        if txn_words[i][3] in pivot_addr_reg_book:
            txn_segment = TxnSegment(int(txn_words[i][3], 16), int(txn_words[i][2], 16), i, file_name)
            expand_line_range(txn_words, i, txn_segment)
            txn_segments.append(txn_segment)

    return txn_segments


def get_words(file_name):
    words = []
    with open(file_name) as fp:
        lines = fp.readlines()

    for line in lines:
        stripped_line = line.strip()
        tmp_words = re.split(r'[ \t\s]\s*', stripped_line)  # use spaces and tabs to split the string
        if tmp_words[0] == "read_reg" or tmp_words[0] == "write_reg":
            tmp_words[-1] = tmp_words[-1].strip('#')  # deannotate the reg address
        words.append(tmp_words)

    return words


def match_txn_segment_in_list(txn_segment, to_be_compared_segments, matched_tags):
    first_matched_found = False
    txn_segment_str = txn_segment.file_name + " line " + str(txn_segment.core_line_num) + " (" + \
                      str(hex(txn_segment.core_reg)) + " = " + str(hex(txn_segment.core_reg_val)) + ")"

    for (i, potential_seg) in enumerate(to_be_compared_segments):
        if txn_segment.other_reg_mapping == potential_seg.other_reg_mapping and \
                len(txn_segment.addr_reg_mapping) == len(potential_seg.addr_reg_mapping):

            print(txn_segment_str + " potentially matches " + potential_seg.file_name +
                  " line " + str(potential_seg.core_line_num) + " (" + str(hex(potential_seg.core_reg)) + " = " +
                  str(hex(potential_seg.core_reg_val)) + ")\n")

            if first_matched_found or matched_tags[i]:
                continue

            first_matched_found = True
            matched_tags[i] = True
            print("***************\nMATCH: " + txn_segment_str + " WITH " + potential_seg.file_name +
                  " line " + str(potential_seg.core_line_num) + " (" + str(hex(potential_seg.core_reg)) + " = " +
                  str(hex(potential_seg.core_reg_val)) + ")\n***************\n")

            txn_segment.matched_target = potential_seg

    if not first_matched_found:
        print(txn_segment_str + " does not match anything, abort...")
        # exit(1)


def match_lists_with_list(txn_segment_lists, to_be_compared_segments):
    matched_tags = [False for _ in range(len(to_be_compared_segments))]
    # first check number of total txns
    num_partitioned_txn = 0
    for txn_segment_list in txn_segment_lists:
        num_partitioned_txn += len(txn_segment_list)

    if num_partitioned_txn != len(to_be_compared_segments):
        print("number of txns does not match!")
        # exit(1)
    else:
        print("number of txns match")

    for txn_segment_list in txn_segment_lists:
        for txn_segment in txn_segment_list:
            match_txn_segment_in_list(txn_segment, to_be_compared_segments, matched_tags)

    # check matched_tags whether all txn_segments have been mapped
    for i in range(len(to_be_compared_segments)):
        if not matched_tags[i]:
            print("line " + str(to_be_compared_segments[i].core_line_num) + " (" +
                  str(hex(to_be_compared_segments[i].core_reg)) + " = " +
                  str(hex(to_be_compared_segments[i].core_reg_val)) + ") has not been mapped")


def txn_segment2graph(txn_segments, addr_analyzer):
    # extract info about addresses in txn_segments into addr_analyzer
    for txn_segment in txn_segments:
        for reg in addr_reg_book:
            if reg in txn_segment.addr_reg_mapping:
                addr = txn_segment.addr_reg_mapping[reg]
                if addr not in addr_analyzer:
                    addr_analyzer[addr] = AddrUsageDesc(addr)

                if reg in input_addr_regs:
                    addr_analyzer[addr].add_consumer(txn_segment)
                elif reg in output_addr_regs:
                    addr_analyzer[addr].add_producer(txn_segment)

    g = nx.DiGraph()
    pos = {}

    for i, txn_segment in enumerate(txn_segments):
        g.add_nodes_from([str(txn_segment.core_line_num)])

    for (addr, addr_usage_desc) in addr_analyzer.items():
        if not addr_usage_desc.is_internal_data_addr():
            print("addr " + str(hex(addr)) + " is not an internal edge")
            continue

        producer_name = str(addr_usage_desc.producers[0].core_line_num)
        # assume only an address only has one producer
        for consumer in addr_usage_desc.consumers:
            consumer_name = str(consumer.core_line_num)
            print("addr " + str(hex(addr)) + " is an edge from " + producer_name + " to " + consumer_name)
            g.add_edges_from([(producer_name, consumer_name)], weight=addr)

    edge_labels = dict([((u, v), str(hex(d['weight']))) for u, v, d in g.edges(data=True)])
    pos = nx.spring_layout(g)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    nx.draw_networkx(g, pos=pos, arrows=True, node_shape='s', node_color='red')
    pylab.show()


def main():
    src_txn_file = sys.argv[1]
    src_words = get_words(src_txn_file)

    src_txn_segments = get_txn_segments(src_words, src_txn_file)
    print("src_txn_segments=", len(src_txn_segments))

    src_addr_analyzer = dict()
    # txn_segment2graph(src_txn_segments, src_addr_analyzer)

    pipeline_txn_files = sys.argv[2:]
    pipeline_stage_num = len(pipeline_txn_files)
    pipeline_txn_words = [get_words(file) for file in pipeline_txn_files]

    pipeline_txn_segments = []
    for i in range(pipeline_stage_num):
        pipeline_txn_segments.append(get_txn_segments(pipeline_txn_words[i], pipeline_txn_files[i]))
        print(len(pipeline_txn_segments[-1]))

    match_lists_with_list(pipeline_txn_segments, src_txn_segments)


if __name__ == '__main__':
    main()
