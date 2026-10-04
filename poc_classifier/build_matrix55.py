"""Writes data/test_matrix55.jsonl. Gold = what the user SHOULD get (assigned
before any pipeline output was looked at). Queries read from the earlier
run_matrix.py (extracted to results/_matrix_queries_raw.json)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
Q = json.load(open(os.path.join(HERE, "results", "_matrix_queries_raw.json"), encoding="utf-8"))
assert len(Q) == 55 and Q[0] == "find me a room" and Q[54] == "2bhk bangalore 15k 2bhk bangalore 15k"
# (ctx, gold, flag_reason or "")
G = [
 (0,"NEW_SEARCH","?: no city given - a search intent, but one could argue UNCLEAR/needs clarification"),
 (0,"NEW_SEARCH",""),(0,"NEW_SEARCH",""),(0,"NEW_SEARCH",""),
 (0,"NEW_SEARCH","?: 'for students' could be read as STUDENT_SUITABILITY, but no rooms shown yet"),
 (0,"NEW_SEARCH",""),(0,"NEW_SEARCH",""),(0,"NEW_SEARCH",""),(0,"NEW_SEARCH",""),(0,"NEW_SEARCH",""),
 (1,"AMENITY",""),
 (1,"ROOM_DETAILS",""),
 (1,"ROOM_PRICE",""),
 (1,"ROOM_DETAILS","?: availability question; no AVAILABILITY label so ROOM_DETAILS is closest"),
 (1,"ROOM_DETAILS","?: 'details on all of them' could be ROOM_TABLE"),
 (1,"LOCATION",""),
 (1,"FAMILY_SUITABILITY","?: could be a REFINEMENT/new family filter"),
 (1,"STUDENT_SUITABILITY","?: 'what about students' could be REFINEMENT (re-filter by students)"),
 (1,"COMPARE_ROOMS",""),
 (1,"ROOM_TABLE",""),
 (1,"UNCLEAR","?: hesitation; could be ACKNOWLEDGMENT"),
 (1,"SHOW_MORE_OPTIONS","?: 'maybe' hedges; could be UNCLEAR"),
 (1,"SHOW_MORE_OPTIONS",""),
 (1,"REJECT_CURRENT",""),
 (1,"SHOW_MORE_OPTIONS",""),
 (1,"SHOW_MORE_OPTIONS","?: 'better' could imply REJECT_CURRENT or REFINEMENT"),
 (1,"REFINEMENT","?: 'cheap ones' could be LOWEST_PRICE (cheapest of shown) or NEW_SEARCH"),
 (1,"REFINEMENT",""),
 (1,"REFINEMENT","?: could be SHOW_MORE_OPTIONS (no concrete area given)"),
 (1,"REFINEMENT","?: could be LOCATION (asking about HSR) or NEW_SEARCH"),
 (1,"NEW_SEARCH","?: restates city + wants cheap rooms mid-conversation; could be REFINEMENT"),
 (1,"ACKNOWLEDGMENT","?: could mean choosing a room (ROOM_DETAILS)"),
 (1,"ROOM_DETAILS","?: 'dusra option' = second option (ROOM_DETAILS) or 'another option' (SHOW_MORE_OPTIONS)"),
 (1,"ROOM_PRICE",""),
 (1,"SHOW_MORE_OPTIONS",""),
 (0,"NEW_SEARCH",""),(0,"NEW_SEARCH",""),
 (0,"NEW_SEARCH","?: heavily garbled ('ne' = any) but still a room search"),
 (1,"LOWEST_PRICE",""),
 (1,"CLOSING","?: thanks + bye; could be ACKNOWLEDGMENT"),
 (0,"NEW_SEARCH","?: keyboard mash + 'rooms pls'; could be UNCLEAR"),
 (0,"NEW_SEARCH","?: truncated ('under' with no number) but intent is clear"),
 (0,"NEW_SEARCH",""),
 (0,"UNCLEAR","?: whitespace only; could be OUT_OF_SCOPE (input rejected)"),
 (0,"UNCLEAR",""),
 (0,"NEW_SEARCH","?: bare repeated keyword; could be UNCLEAR"),
 (0,"NEW_SEARCH",""),(0,"NEW_SEARCH",""),
 (0,"NEW_SEARCH","?: emoji + city only; could be UNCLEAR"),
 (1,"LOWEST_PRICE",""),
 (0,"NEW_SEARCH",""),
 (0,"NEW_SEARCH","?: angry complaint + room request; could be UNCLEAR/out of scope"),
 (0,"NEW_SEARCH",""),
 (0,"NEW_SEARCH","?: vague ('anything?'); could be UNCLEAR"),
 (0,"NEW_SEARCH",""),
]
assert len(G) == 55
with open(os.path.join(HERE,"data","test_matrix55.jsonl"),"w",encoding="utf-8") as f:
    for i,(q,(c,l,fl)) in enumerate(zip(Q,G),1):
        f.write(json.dumps({"id":i,"text":q,"label":l,"has_room_context":bool(c),"debatable":fl},ensure_ascii=False)+"\n")
print("ok")
