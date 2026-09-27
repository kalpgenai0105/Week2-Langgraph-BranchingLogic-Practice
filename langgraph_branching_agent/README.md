# LangGraph Branching Agent

Classifier node routes a topic to a Technical or Business report node.

## Run
```
pip install -r requirements.txt
copy .env.example .env   # add your GOOGLE_API_KEY
python agent.py "Kubernetes autoscaling"
python agent.py "Pricing strategy for a SaaS launch"
python agent.py --save "Kubernetes autoscaling"   # also writes reports/technical-kubernetes-autoscaling.md
```
