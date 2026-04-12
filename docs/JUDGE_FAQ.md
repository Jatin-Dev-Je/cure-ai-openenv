# Judge FAQ

## Q1. What practical problem does Cure AI solve?
Cure AI evaluates incident-response agent behavior on realistic outage scenarios, emphasizing safe and useful remediation decisions.

## Q2. Why this domain?
On-call incident response is high impact and underrepresented in agent benchmarks despite clear real-world demand.

## Q3. How is difficulty represented?
Through three deterministic tasks that escalate in operational ambiguity and remediation complexity.

## Q4. How are agents evaluated?
Using deterministic grading signals for diagnosis, fix quality, and root-cause labeling, plus safety penalties for dangerous actions.

## Q5. Is the environment reproducible?
Yes. Task progression and evaluator-facing output contract are deterministic and parser-safe.

## Q6. Is this deployable?
Yes. It supports OpenEnv validate workflows, Dockerized serving, and HF Space deployment.

## Q7. What makes this novel?
It combines practical SRE incidents, safety-aware grading, and reproducible evaluation protocol in one OpenEnv-compatible benchmark.
