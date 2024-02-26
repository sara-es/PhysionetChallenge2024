## Classification

# Update 26th Feb:
- check why the ready-made function to load signal loads the signal in shape (5000, 12) instaed of (12, 5000)?!
- finish args dict for initialisation
- check: what if ECGdatast gets only records as load_signal() function loads the signal AND the corresponding header?!

#### After it works
- right now, demographics for classifcation are sex and age, is it worth adding height and weight?
- metric functions/validation
- dynamically determine number of ECG leads
- refactor train test splot in team_code.py

#### Low priority
- investigate 'Bottleneck' in seresnet18.py