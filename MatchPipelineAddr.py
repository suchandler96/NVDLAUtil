import re
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pylab
from TxnSegment import *


match_dp_ans = [0, None, None]


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


def match_txn_segment_candidates_in_list(txn_segment, to_be_compared_segments):
    txn_segment_str = txn_segment.file_name + " line " + str(txn_segment.core_line_num) + " (" + \
                      str(hex(txn_segment.core_reg)) + " = " + str(hex(txn_segment.core_reg_val)) + ")"

    for (i, potential_seg) in enumerate(to_be_compared_segments):
        if txn_segment.other_reg_mapping == potential_seg.other_reg_mapping and \
                len(txn_segment.addr_reg_mapping) == len(potential_seg.addr_reg_mapping):
            txn_segment.match_candidates.append(i)
            # if uncomment lines below, it will match the txn_segment with the first available candidate seen
            '''
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
            '''
    '''
    if not first_matched_found:
        print(txn_segment_str + " does not match anything...")
        # exit(1)
    '''


def match_dp(txn_segment_lists, txn_to_explore_pos, to_be_compared_segments, matched_tags, existing_addr_mappings):
    if txn_to_explore_pos[0] == len(txn_segment_lists):
        # recursion base
        # count matched src_segments
        matched_src_segments = 0
        for tag in matched_tags:
            if tag:
                matched_src_segments += 1

        if matched_src_segments < match_dp_ans[0]:
            # not as good as previously explored answer
            return
        print("Matched segments: from " + str(match_dp_ans[0]) + " to " + str(matched_src_segments))

        target_lists = []
        for txn_segment_list in txn_segment_lists:
            target_list = []
            for segment in txn_segment_list:
                target_list.append(segment.matched_target)
            target_lists.append(target_lists)
        '''
        for txn_segment_list in txn_segment_lists:
            for txn_segment in txn_segment_list:
                if txn_segment.matched_target is not None:
                    print("MATCH: " + txn_segment.desc_str() + " WITH " +
                          to_be_compared_segments[txn_segment.matched_target].desc_str() + "\n")
                else:
                    if len(txn_segment.match_candidates) == 0:
                        print(txn_segment.desc_str() + " doesn't match any src txn segments because NO CANDIDATES\n")
                    else:
                        print(txn_segment.desc_str() + " doesn't match any src txn segments because NOT SUITABLE\n")
        '''
        to_return_mappings = []
        for stage_id, mapping in enumerate(existing_addr_mappings):
            to_return_mappings.append(dict(mapping))
            for key, value in mapping.items():
                print("stage " + str(stage_id) + ": " + str(hex(key)) + "->" + str(hex(value)))
        match_dp_ans[0] = matched_src_segments
        match_dp_ans[1] = target_lists
        match_dp_ans[2] = to_return_mappings
        return

    segment = txn_segment_lists[txn_to_explore_pos[0]][txn_to_explore_pos[1]]
    if len(segment.match_candidates) != 0:
        for candidate_id in segment.match_candidates:
            if matched_tags[candidate_id]:
                continue

            new_addr_mappings = {}

            # check whether this mapping agrees with those in existing_addr_mapping
            is_valid_mapping = True
            for reg, reg_val in segment.addr_reg_mapping.items():
                target_reg_val = to_be_compared_segments[candidate_id].addr_reg_mapping[reg]

                if reg_val not in existing_addr_mappings[txn_to_explore_pos[0]]:
                    new_addr_mappings[reg_val] = target_reg_val
                else:
                    if existing_addr_mappings[txn_to_explore_pos[0]][reg_val] != target_reg_val:
                        # this mapping is not coherent with previous ones, try another candidate
                        is_valid_mapping = False
                        break
                    # else the addr mapping is coherent with previous ones

            if is_valid_mapping:
                # add the newly added mappings
                for new_addr, new_mapped_address in new_addr_mappings.items():
                    existing_addr_mappings[txn_to_explore_pos[0]][new_addr] = new_mapped_address

                segment.matched_target = candidate_id
                matched_tags[candidate_id] = True
                next_pos = [txn_to_explore_pos[0], txn_to_explore_pos[1] + 1]
                if next_pos[1] == len(txn_segment_lists[txn_to_explore_pos[0]]):
                    next_pos = [txn_to_explore_pos[0] + 1, 0]

                match_dp(txn_segment_lists, next_pos, to_be_compared_segments, matched_tags, existing_addr_mappings)
                #if answer is not None:      # report the first found answer
                    #return answer

                # not returning means this candidate is not ok, clean up the changed variables
                segment.matched_target = None
                matched_tags[candidate_id] = False
                for key in new_addr_mappings:
                    existing_addr_mappings[txn_to_explore_pos[0]].pop(key)

    # this segment is not matching any candidates (or no candidates at all). skip this segment
    next_pos = [txn_to_explore_pos[0], txn_to_explore_pos[1] + 1]
    if next_pos[1] == len(txn_segment_lists[txn_to_explore_pos[0]]):
        next_pos = [txn_to_explore_pos[0] + 1, 0]
    match_dp(txn_segment_lists, next_pos, to_be_compared_segments, matched_tags, existing_addr_mappings)


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

    # find all candidates for each txn_segment
    for txn_segment_list in txn_segment_lists:
        for txn_segment in txn_segment_list:
            match_txn_segment_candidates_in_list(txn_segment, to_be_compared_segments)

    # explore possible matching with DP
    # can add knowledge from mem trace matching here
    existing_addr_mappings = [dict() for _ in range(len(txn_segment_lists))]
    match_dp(txn_segment_lists, [0, 0], to_be_compared_segments, matched_tags, existing_addr_mappings)

    # check matched_tags whether all txn_segments have been mapped
    for i in range(len(to_be_compared_segments)):
        if not matched_tags[i]:
            print("line " + str(to_be_compared_segments[i].core_line_num) + " (" +
                  str(hex(to_be_compared_segments[i].core_reg)) + " = " +
                  str(hex(to_be_compared_segments[i].core_reg_val)) + ") has not been mapped")


def txn_segment2graph(txn_segments, addr_analyzer):
    # extract info about addresses in txn_segments into addr_analyzer
    for txn_segment in txn_segments:
        for reg in addr_reg_list:
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
