# This program is to generate a search markdown table for helena to scrape

# Define the terms to search
Energy_term_list = ["Wind", "Solar", "Power", "Energy", "Generator", "Coal",\
                    "Oil", "Natural Gas", "Geothermal", "Hydropower"]

ML_term_list = ["machine learning", "deep learning", "support vector machine", \
                "random forest", "regression tree", "neural network"]

RS_term_list = ["remote sensing", "satellite", "aerial", "UAV",\
                "unmanned aerial vehicle", "hyperspectral"]

file_name = 'table.txt'

if __name__ == '__main__':
    with open(file_name, 'w') as f:
        for i1, energy_term in enumerate(Energy_term_list):
            for i2, ml_term in enumerate(ML_term_list):
                for i3, rs_term in enumerate(RS_term_list):
                    index_str =  str(i1 + 1) + "-" + str(i2 + 1) + "-" + str(i3 + 1)
                    link = "(https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C34&q=%22"
                    link += energy_term.replace(" ", "+")
                    link += "%22%3B+%22"
                    link += ml_term.replace(" ", "+")
                    link += "%22%3B+%22"
                    link += rs_term.replace(" ", "+")
                    link += "%22&btnG=)"
                    f.write("|" + index_str + " | " + energy_term + " | " + ml_term + " | " \
                            + rs_term + " | " + "[link]" + link + " |\n")






