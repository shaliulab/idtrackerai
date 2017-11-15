class BkgSubtraction(BoxLayout):
    def __init__(self, **kwargs):
        super(BkgSubtraction, self).__init__(**kwargs)
        self.bkg = None
        #set useful popups
        #saving:
        self.saving_popup = Popup(title='Saving',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.saving_popup.bind(on_open=self.save_bkg)
        #computing:
        self.computing_popup = Popup(title='Computing',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.computing_popup.bind(on_open=self.compute_bkg)
        global CHOSEN_VIDEO

    def subtract_bkg(self, *args):
        if hasattr(CHOSEN_VIDEO.old_video, "bkg") or hasattr(CHOSEN_VIDEO.video, "bkg"):
            if CHOSEN_VIDEO.old_video.bkg is not None:
                self.bkg = CHOSEN_VIDEO.old_video.bkg
            elif CHOSEN_VIDEO.video.bkg is not None:
                self.bkg = CHOSEN_VIDEO.video.bkg
        else:
            self.compute_bkg()

    def save_bkg(self, *args):
        CHOSEN_VIDEO.video.bkg = self.bkg
        CHOSEN_VIDEO.video.save()
        self.saving_popup.dismiss()

    def compute_bkg(self, *args):
        self.bkg = computeBkg(CHOSEN_VIDEO.video)
        self.save_bkg()
        self.computing_popup.dismiss()
