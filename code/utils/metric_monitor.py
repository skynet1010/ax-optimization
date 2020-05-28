def calc_metrics(m):
    m["sensitivity"] = m["TP"]/max(1,(m["TP"]+m["FN"]))
    m["miss_rate"] = 1-m["sensitivity"]
    m["specificity"] = m["TN"]/max(1,(m["TN"]+m["FP"]))
    m["fallout"] = 1-m["specificity"]

    m["precision"] = m["TP"]/max(1,(m["TP"]+m["FP"]))
    m["NPV"] = m["TN"]/max(1,(m["TN"]+m["FN"]))#negative_predictive_value

    m["F1"] = 2*m["precision"]*m["sensitivity"]/max(1,(m["sensitivity"]+m["precision"]))

    return m