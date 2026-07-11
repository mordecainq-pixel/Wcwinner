"""WNBA game prediction system: self-calculated Elo blended into a fitted
offensive/defensive rating margin model (basketball's answer to the Dixon-
Coles Poisson model used for soccer in wcwinner/ -- goals-as-Poisson-events
doesn't fit a sport where teams score 70-90+ points, so the core engine here
predicts point margin and total directly via a Normal-distribution fit
instead). Shares the Bet Builder / validation architecture with wcwinner/
since those are sport-agnostic.
"""
