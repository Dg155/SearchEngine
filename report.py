import shelve

def generate_report(title, stats, file):
    f.write(f"----- {title} -----\n")
    if type(stats) == dict:
        for key, value in stats.items():
            f.write(f"{key} {value}\n")
    elif type(stats) == list or type(stats) == set:
        for value in stats:
            f.write(f"{value}\n")
    elif type(stats) == int or type(stats) == str or type(stats) == bool or type(stats) == float:
        f.write(f"{stats}\n")
    f.write("------------------\n")


if __name__ == "__main__":
    with open("report.txt", "w") as f:

        with shelve.open("analystInfo.shelve") as analystInfo:
            generate_report("Indexed Documents", analystInfo["indexedDocumesnts"], f)
            generate_report("Unique Tokens", analystInfo["uniqueTokens"], f)
            generate_report("Total Size", analystInfo["kilobytes"], f)
            generate_report("Skipped Documents", analystInfo["skippedDocuments"], f)
    
        