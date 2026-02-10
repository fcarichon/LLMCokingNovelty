from Novelty import Newness, Uniqueness, Difference
from Surprise import Surprise
from utils import pmi, data_analysis, docs_distribution

def compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, types, thresholds):

    saved_results = {}
    newness_ = Newness(KB_dist, variation_dist)
    uniqueness = Uniqueness(KB_dist)
    difference = Difference(KB_matrix, variation_dist, N=thresholds["neighbors"])
    surprise = Surprise(New_EB_PMI)
    
    if types["newness_div"]:
        saved_results["newness_div"], saved_results["novelty_newdiv"] = newness_.divergent_terms(thr_div=thresholds["newness_div"], thr_new=thresholds["novelty_new"])
    if types["newness_prob"]:
        saved_results["newness_prob"], saved_results["novelty_newprob"] = newness_.probable_terms(thr_prob=thresholds["newness_prob"], thr_new=thresholds["novelty_new"])
    if types["new_extremes"]:
        saved_results["newness_extr"], saved_results["novelty_newextr"] = newness_.new_extremes(thr_div=thresholds["newness_div"], thr_new=thresholds["novelty_new"], pareto=0.8)
    if types["new_rank"]:
        saved_results["newness_rank"], saved_results["novelty_rank"] = newness_.new_ranking(thr_new=thresholds["novelty_new"], rank_thr=int(len(variation_dist)/thresholds["newness_rankm"]))
  
    if types["uniq_dist"]:
        saved_results["uniq_dist"], saved_results["novelty_uniqdist"] = uniqueness.dist_to_proto(variation_dist, thr_uniq=thresholds["uniq_dist"])
    if types["uniq_proto"]:
        saved_results["uniq_proto"], saved_results["novelty_uniqproto"] = uniqueness.proto_dist_shift(NewKB_dist, thr_uniqp=thresholds["uniq_proto"])  # Devrait être 

    if types["diff_global"]:
        if neighbor_dist==0.:
            neighbor_dist = difference.dist_estimate()
        saved_results["diff_global"], saved_results["nolvety_diffglob"] = difference.ratio_to_all(neighbor_dist, thr_diff=thresholds["diff_global"])
    if types["diff_local"]:
        if neighbor_dist==0.:
            neighbor_dist = difference.dist_estimate()
        saved_results["diff_local"], saved_results["nolvety_diffloc"] = difference.ratio_to_neighbors(neighbor_dist, thr_diff=thresholds["diff_local"])

    if types["new_surprise"]:
        saved_results["new_surprise"], saved_results["nolvety_newsurpr"] = surprise.new_surprise(EB_PMI, thr_surp=thresholds["new_surprise"])
    if types["dist_surprise"]:
        saved_results["dist_surprise"], saved_results["nolvety_distsurpr"] = surprise.uniq_surprise(dict_know_pmi, eps= 0.00, thr_surp=thresholds["dist_surprise"])

    return saved_results


