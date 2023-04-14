#!/usr/bin/env python3
"""
Requires:
 - addgene fastas
 - blast database of addgene fastas
 - blastn in the environment
"""
import argparse
import subprocess
import warnings
from itertools import combinations
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from Bio import BiopythonWarning, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

pd.options.mode.chained_assignment = None


def bash(inCommand, autoOutfile=True):
    if autoOutfile:
        tmp = NamedTemporaryFile()
        subprocess.call(inCommand + " > " + tmp.name, shell=True)
        f = open(tmp.name, "r")
        tmp.close()
        return f.read()
    else:
        subprocess.call(inCommand, shell=True)


def get_gbk(addgeneNum):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        path = f"./in_data/FASTAs_2021-08-09/{addgeneNum}.fasta"
        with open(path, "r") as file_handle:
            gbk = list(SeqIO.parse(file_handle, "fasta"))[0]
            return gbk


def pairwise_BLAST(
    addgeneNum1,
    addgeneNum2,
    returnRaw=False,
    perc_identity=98,
    penalty=-8,
    reward=2,
    gapopen=4,
    gapextend=6,
):
    q = NamedTemporaryFile()
    s = NamedTemporaryFile()

    gbk1 = get_gbk(addgeneNum1)
    gbk2 = get_gbk(addgeneNum2)

    # genome is *2 becuase this solves origin issues (with filtering)
    SeqIO.write(SeqRecord(gbk1.seq * 2, id=str(addgeneNum1)), q.name, "fasta")
    SeqIO.write(SeqRecord(gbk2.seq * 2, id=str(addgeneNum2)), s.name, "fasta")

    # actual pairwise BLAST
    flags = (
        "qseqid sseqid pident length qlen slen qstart qend sstart send sseq qseq sframe"
    )
    extras = f"-culling_limit 1 -dust no -soft_masking false -penalty {penalty} -reward {reward} -perc_identity {perc_identity} -gapopen {gapopen} -gapextend {gapextend}"
    out = bash(f'blastn -query {q.name} -subject {s.name} {extras} -outfmt "6 {flags}"')
    q.close()
    s.close()

    # organizes hits into a dataframe,fixes plas len, filters sorts
    out = [ele.split("\t") for ele in out.split("\n") if ele]
    out = pd.DataFrame(out, columns=flags.split())
    out = out.apply(pd.to_numeric, errors="ignore")
    out["qlen"] = (out["qlen"] / 2).astype("int32")
    out["slen"] = (out["slen"] / 2).astype("int32")
    out = out[out["pident"] > 80]  # this is conservative --change maybe
    out = out.sort_values(by="length", ascending=False)  # could sort different ways

    # swaps the sstart and send if sstrand==-1
    sstartSwap = np.where(out["sframe"] == -1, out["send"], out["sstart"])
    sendSwap = np.where(out["sframe"] == -1, out["sstart"], out["send"])
    out["sstart"] = sstartSwap
    out["send"] = sendSwap

    # this adds the plas len to all positions below the plas len
    # this allows for correct filtering of overlapping sequences
    # and also solves the origin issue
    for sORq in ["s", "q"]:
        start = np.where(
            (out[f"{sORq}start"] < out[f"{sORq}len"])
            & (out[f"{sORq}end"] < out[f"{sORq}len"]),
            out[f"{sORq}start"] + out[f"{sORq}len"],
            out[f"{sORq}start"],
        )
        end = np.where(
            (out[f"{sORq}start"] < out[f"{sORq}len"])
            & (out[f"{sORq}end"] < out[f"{sORq}len"]),
            out[f"{sORq}end"] + out[f"{sORq}len"],
            out[f"{sORq}end"],
        )
        out[f"{sORq}start"] = start
        out[f"{sORq}end"] = end

    # this removes overlaps
    out["remove"] = False
    for i, j in combinations(out.index, 2):
        if not out.loc[j, "remove"]:
            out.loc[j, "remove"] = (
                (out.loc[j, "sstart"] >= out.loc[i, "sstart"])
                and (out.loc[j, "send"] <= out.loc[i, "send"])
            ) | (
                (out.loc[j, "qstart"] >= out.loc[i, "qstart"])
                and (out.loc[j, "qend"] <= out.loc[i, "qend"])
            )
    out = out.loc[~out["remove"]]
    out = out.drop(columns="remove")

    # this subtracts position values so they are all under plasmid len
    for sORq in ["s", "q"]:
        for startORend in ["start", "end"]:
            out[f"{sORq}{startORend}"] = np.where(
                out[f"{sORq}{startORend}"] > out[f"{sORq}len"],
                out[f"{sORq}{startORend}"] - out[f"{sORq}len"],
                out[f"{sORq}{startORend}"],
            )

    # this drops all hits that are BOTH greater than plas len -- ie all duplicates
    out = out.drop(
        out[(out["sstart"] > out["slen"]) & (out["send"] > out["slen"])].index
    )
    out = out.drop(
        out[(out["qstart"] > out["qlen"]) & (out["qend"] > out["qlen"])].index
    )

    # probably doesn't do anything, just for safety
    out = out.drop_duplicates(
        subset=[
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "qstart",
            "qend",
            "qlen",
            "sstart",
            "send",
            "slen",
            "sframe",
        ]
    )

    # this "divides" sequences that are near matches -- if sequences are
    # almost 100% identical, the hit will span both 'copies' that were
    # generated to deal with the origin splitting issue, and needs to be
    # split in half
    for index in out.index:
        curRow = out.loc[index]
        if curRow["length"] > curRow["slen"] or curRow["length"] > curRow["qlen"]:
            sseq = curRow["sseq"][0 : int(len(curRow["sseq"]) / 2)]
            out.at[index, "sseq"] = sseq

            qseq = curRow["qseq"][0 : int(len(curRow["qseq"]) / 2)]
            out.at[index, "qseq"] = qseq

            out.at[index, "length"] = curRow["length"] / 2

    out = out.drop_duplicates()
    out = out.reset_index(drop=True)
    out["qperc"] = 100 * out["length"] / out["qlen"]
    out["sperc"] = 100 * out["length"] / out["slen"]
    return out


def BLAST_all_addgene(
    inSeq, perc_identity=98, penalty=-8, reward=2, gapopen=4, gapextend=6
):
    inSeq = inSeq.replace("-", "")
    q = NamedTemporaryFile()

    # genome is *2 becuase this solves origin issues (with filtering)
    SeqIO.write(SeqRecord(Seq(inSeq), id="temp"), q.name, "fasta")

    dbPath = "./in_data/BLAST_db_2021-08-09/plasmid_sequences_2021-09-08.fasta"

    # actual pairwise BLAST
    flags = (
        "qseqid sseqid pident length qlen slen qstart qend sstart send sseq qseq sframe"
    )

    extras = f"-max_target_seqs 51384 -evalue .00001 -num_threads 2 -word_size 28 -penalty {penalty} -reward {reward} -perc_identity {perc_identity} -gapopen {gapopen} -gapextend {gapextend}"
    out = bash(f'blastn -query {q.name} -db {dbPath} {extras} -outfmt "6 {flags}"')

    q.close()

    # organizes hits into a dataframe,fixes plas len, filters sorts
    out = [ele.split("\t") for ele in out.split("\n") if ele]
    out = pd.DataFrame(out, columns=flags.split())
    out = out.apply(pd.to_numeric, errors="ignore")

    out["qlen"] = (out["qlen"] / 2).astype("int32")
    out["slen"] = (out["slen"] / 2).astype("int32")
    out = out[out["length"] > len(inSeq) - 10]  # this is conservative --change maybe
    out = out[out["length"] < len(inSeq) + 10]
    out = out.sort_values(by="length", ascending=False)  # could sort different ways

    # this drops all hits that are BOTH greater than plas len -- ie all duplicates
    out = out.drop(
        out[(out["sstart"] >= out["slen"]) & (out["send"] >= out["slen"])].index
    )
    out = out.drop(
        out[(out["qstart"] >= out["qlen"]) & (out["qend"] >= out["qlen"])].index
    )

    # swaps the sstart and send if sstrand==-1
    sstartSwap = np.where(out["sframe"] == -1, out["send"], out["sstart"])
    sendSwap = np.where(out["sframe"] == -1, out["sstart"], out["send"])
    out["sstart"] = sstartSwap
    out["send"] = sendSwap

    return out


def main():
    parser = argparse.ArgumentParser(
        description="does pairwise idf scores for plasmids"
    )
    parser.add_argument("-i", "--i", help="i position in df", required=True)
    parser.add_argument("-j", "--j", help="j position in df", required=True)
    parser.add_argument(
        "-1", "--id1", help="addgene id number for first plasmid", required=True
    )
    parser.add_argument(
        "-2", "--id2", help="addgene id number for second plasmid", required=True
    )
    parser.add_argument(
        "-v", "--variant", help="ColE1 variant num in df", required=True
    )

    args = parser.parse_args()
    i = int(args.i)
    j = int(args.j)
    variant = int(args.variant)

    plasCount = 51384

    c = []
    pairwiseDF = pairwise_BLAST(args.id1, args.id2)

    for x in range(len(pairwiseDF)):

        hit = BLAST_all_addgene(pairwiseDF.loc[x]["qseq"])

        a = set(hit["sseqid"])
        numHits = len(a)

        c.append(numHits)

    if len(pairwiseDF) == 0:
        c = [plasCount]  # default case if no shared sequence

    c = [plasCount / ele for ele in c]
    c = sorted(c)
    baselineScore = c.pop()
    if c:
        extraScore = sum(c) / (len(c) + 1)
    else:
        extraScore = 0

    finalScore = baselineScore + extraScore
    if finalScore > plasCount / 2:
        finalScore = plasCount / 2

    # multiplies by ave percent of bases of plasmids shared
    # adds one for log so no div by 0 error below
    finalScore = (
        finalScore * ((sum(pairwiseDF["qperc"]) + sum(pairwiseDF["sperc"])) / 200)
    ) + 1
    pi = round((sum(pairwiseDF["qperc"]) + sum(pairwiseDF["sperc"])) / 200, 5)
    print(variant, i, j, args.id1, args.id2, np.log(finalScore), pi, sep=",")


if __name__ == "__main__":
    main()
