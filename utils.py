### Function to disable/enable resiter button
def switch(event,txt,txt2,takeImg):
    if len(txt.get())==0 or len(txt2.get())==0:
        takeImg['state']='disabled'
    else:
        takeImg['state']='normal'

    