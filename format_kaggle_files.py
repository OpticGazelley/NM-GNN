import datatable as dt
from datetime import datetime, timezone
from datatable import f,join,sort
import sys
import os

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
raw_name = os.path.basename(inPath)
dataset_name = {
    'HI-Small_Trans.csv': 'Small_HI',
    'LI-Small_Trans.csv': 'Small_LI',
    'HI-Medium_Trans.csv': 'Medium_HI',
    'LI-Medium_Trans.csv': 'Medium_LI',
}.get(raw_name, os.path.splitext(raw_name)[0])
out_dir = os.path.join(os.path.dirname(inPath), dataset_name)
os.makedirs(out_dir, exist_ok=True)
outPath = os.path.join(out_dir, "formatted_transactions.csv")

raw = dt.fread(inPath, columns = dt.str32)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,Is Laundering\n"

with open(outPath, 'w') as writer:
    writer.write(header)
    for i in range(raw.nrows):
        datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M').replace(tzinfo=timezone.utc)
        ts = int(datetime_object.timestamp())

        cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
        cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

        fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

        fromAccIdStr = raw[i,"From Bank"] + "|" + raw[i,2]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw[i,"To Bank"] + "|" + raw[i,4]
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedOrig = float(raw[i,"Amount Received"])
        amountPaidOrig = float(raw[i,"Amount Paid"])

        isl = int(raw[i,"Is Laundering"])

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

        writer.write(line)

formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPath)
