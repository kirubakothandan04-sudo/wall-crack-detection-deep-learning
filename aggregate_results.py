import json, glob

rows = []
for f in sorted(glob.glob("logs/*_log.json")):
    variant = f.split("/")[-1].replace("_log.json", "")
    d = json.load(open(f))["test"]
    rows.append((variant, d["accuracy"], d["f1"], d["precision"], d["recall"]))

print(f"{'Variant':<20}{'Acc':<8}{'F1':<8}{'Prec':<8}{'Recall':<8}")
for r in rows:
    print(f"{r[0]:<20}{r[1]:<8.4f}{r[2]:<8.4f}{r[3]:<8.4f}{r[4]:<8.4f}")