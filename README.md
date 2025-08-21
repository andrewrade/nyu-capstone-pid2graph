# NYU Capstone Project 2024: Digitizing P&ID into Industrial Knowledge Graphs

# Team Members:

- Andrew Deur <ad3254@nyu.edu>;
- Manan Prakashkumar Patel <mp6561@nyu.edu>;
- Manan Shah <ms15037@nyu.edu>
- Isha Slavin <ivs225@nyu.edu>

# Mentors:

- Sidhant Krishna
- Rubert Martin
- Shourabh Rawat (shourabh.rawat@symphonyai.com)

# Project:

P&IDs (Piping and Instrumentation Diagrams) are detailed schematic representations used extensively in industrial and manufacturing applications. They provide a comprehensive visual overview of the physical interconnections and control schemes in a process system. P&IDs typically include Piping, Process equipment (e.g. vessels, pumps, heat exchangers), Control devices and instrumentation, Valves and fittings and Safety systems SymphonyAI offers a range of AI-powered solutions for the industrial and manufacturing sector, focusing on improving operational efficiency, asset performance, and decision-making. Symphony AI leverages P&IDs to build a structured knowledge graph representation of the complete installation across a new customer. The manual process of digitizing these types of diagrams is a major roadblock in the process of onboarding new customers and factories. This process often requires manual effort especially when onboarding new customers given the wide spectrum of P&ID schemas across the world. In this project, our aim is to develop \*CV and NLP) algorithms to fully automate the process of ingesting and extracting structured information from P&IDs specs and diagrams and translate them into a knowledge graph comprising of nodes (e.g. pumps) and edges (eg pipes). The approach will leverage both visual and textual information from images and pdf documents to first parse out the P&ID specifications from PDF documents and then use that information to detect components within the P&ID images using computer vision, nlp and OCR. Relevant

# Research and Datasets:

- [Microsoft - Engineering Document (P&ID) Digitization](https://devblogs.microsoft.com/ise/engineering-document-pid-digitization/)
  - [Github Repo](https://github.com/Azure-Samples/digitization-of-piping-and-instrument-diagrams/tree/main)
  - [Github Repo](https://github.com/Azure-Samples/MLOpsManufacturing)
  - [Github - Architecture](https://github.com/Azure-Samples/digitization-of-piping-and-instrument-diagrams/blob/main/docs/architecture.md)
- [End-to-end digitization of image format piping and instrumentation diagrams at an industrially applicable level]()
- [Digitize-PID: Automatic Digitization of Piping and Instrumentation Diagrams](https://arxiv.org/pdf/2109.03794)
- [Digitization of chemical process flow diagrams using deep convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S2772508122000631?via%3Dihub)
- [P&ID Symbols and Meanings](https://www.edrawmax.com/article/p-and-id-symbols.html)
- Automatic Digitization of Engineering Diagrams using Deep Learning and Graph Search [Video](https://youtu.be/GX9_BDeumN0) [Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w8/Mani_Automatic_Digitization_of_Engineering_Diagrams_Using_Deep_Learning_and_Graph_CVPRW_2020_paper.pdf) [Slides](https://cvpr-dira.lipingyang.org/wp-content/uploads/2020/06/Mani_7-slides.pdf)
- [Symphony AI - Industrial AI Knowledge Graphs Asset Hierarchy](https://www.symphonyai.com/resources/blog/industrial/industrial-ai-knowledge-graphs-asset-hierarchy/)
- [Capture the intelligence locked in Piping and Instrumentation Diagrams (P&IDs)](https://www.symphonyai.com/resources/blog/industrial/capture-intelligence-piping-instrumentation-diagrams/)
- [piping-instrumentation-diagrams-ingestion](https://www.symphonyai.com/industrial/piping-instrumentation-diagrams-ingestion/)

# Data:

For this project, team will be expected to leverage both public as well as symphony ai specific datasets to develop their solutions as well as prove out the efficacy and generalizability of their solutions.

Here are the publicly available synthetic datasets that will be considered for initial development.

- [Digitize-PID: Automatic Digitization of Piping and Instrumentation Diagrams](https://arxiv.org/pdf/2109.03794)
  - [Dataset](https://drive.google.com/drive/u/0/folders/1gMm_YKBZtXB3qUKUpI-LF1HE_MgzwfeR)
- [how-read-oil-and-gas-pid-symbols](https://kimray.com/training/how-read-oil-and-gas-pid-symbols)
  - [how-read-oil-and-gas-pid-symbols-pdf](https://kimray.com/sites/default/files/uploads/training-demos/Kimray%20How%20to%20Read%20an%20Oil%20%26%20Gas%20P%26ID%20Reference%20Guide.pdf)
- [P&ID Symbols and Meanings](https://www.edrawmax.com/article/p-and-id-symbols.html)


Team will be encouraged to do their literature review to identify or build additional datasets as needed. The team will also have the opportunity to work with mentors to have their solutions run against the Symphony AI data to test the generalizability of their solutions.

# Communication Channel:

[Slack](https://app.slack.com/client/T07Q57KGTL0/C07P1SXV78F?ssb_vid=.d0aa27f08318603994461fc0fed53903)

# Regular Meetings:

Friday, Afternoon PT On Teams (2 PM PT)
Zoom

# Progress, Goals and Timeline:

Week 0:

Milestone 1: Do a presentation next Friday on these topics

- Literature Review: approaches/papers/ methodologies. Come up with recommendation on potential approaches to tackle this problem.
- Data Analysis: Also highlight existing approaches and results.
- Tools and Libraries:
- Understand the key challenges
- Things that should be clearly understood: relevant offline metrics, data sources, start with a current model that is known to work.

Stretch Goal:
• Create a baseline on the dataset

Week 1:

Team Updates:
Add async updates from each indivdual member: Highlight what you worked on and blockers

- Andrew Deur <ad3254@nyu.edu>
- Manan Prakashkumar Patel <mp6561@nyu.edu>
- Manan Shah <ms15037@nyu.edu>
- Isha Slavin <ivs225@nyu.edu>

Inputs from the Symphony AI Team 
4 core problems 
- Improving the overall accuracy of the system by leveraging complimentary multimodal information (ocr, visual, text, semantics, knowledge base, graph connectedness, distributional semantics)
- Improving the generalizability to new domains (when all symbols are not known at training time)
- Improving the mapping to related structured sources
- Able to reason over this graph


Week 2:

Milestone 2:
- Build baselines
  - Object Detection and Classifiocation on Dataset 1 (50+) [Andrew]
    - recommendation on the core alogirhtn (yolo)   
  - Baseline on text recognition on dataset 1 [Manan S]
    - recommendation on the OCR (oss or commercial)
  - Baseline for edge detection on dataset 1 [Isha]
    - recommendation on best approach to take
    - (stretch) edge classification [can leverage learning from object detection/classification]
  - Baseline on the graph construction [Manan]
    - construct a graph (based on existing ground truth on dataset 1 with noise injected)      

Week 3:

Week 4:
