from __future__ import absolute_import, division, print_function
import kivy
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.graphics import *
from kivy.graphics.transformation import Matrix
from idtrackerai.gui.visualise_video import VisualiseVideo
from idtrackerai.gui.kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
import matplotlib
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
import cv2
from pprint import pprint
from idtrackerai.video import Video
from idtrackerai.utils.py_utils import  getExistentFiles, get_spaced_colors_util
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.groundtruth_utils.generate_individual_groundtruth import generate_individual_groundtruth
from idtrackerai.groundtruth_utils.compute_individual_groundtruth_statistics import get_individual_accuracy_wrt_groundtruth

class IndividualValidator(BoxLayout):
    def __init__(self,
                chosen_video = None,
                deactivate_validation = None,
                **kwargs):
        super(IndividualValidator, self).__init__(**kwargs)
        global CHOSEN_VIDEO, DEACTIVATE_VALIDATION
        CHOSEN_VIDEO = chosen_video
        DEACTIVATE_VALIDATION = deactivate_validation
        self.with_gaps = True
        self.visualiser = VisualiseVideo(chosen_video = CHOSEN_VIDEO)
        self.warning_popup = Popup(title = 'Warning',
                            content = CustomLabel(text = 'The video has not been tracked yet. Track it before performing validation.'),
                            size_hint = (.3,.3))
        self.loading_popup = Popup(title='Loading',
            content=CustomLabel(text='wait ...'),
            size_hint=(.3,.3))

        self.popup_saving = Popup(title='Saving',
            content=CustomLabel(text='wait ...'),
            size_hint=(.3,.3))

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def show_saving(self, *args):
        self.popup_saving.open()

    def create_choose_list_of_blobs_popup(self):
        self.lob_container = BoxLayout()
        self.lob_box = BoxLayout(orientation="vertical")
        self.lob_label = CustomLabel(text='We detected two different trajectory files. Which one do you want to use for individual validation?')
        self.lob_btns_container = BoxLayout()
        self.lob_btn1 = Button(text = "With gaps")
        self.lob_btn2 = Button(text = "Without gaps")
        self.lob_btns_container.add_widget(self.lob_btn1)
        self.lob_btns_container.add_widget(self.lob_btn2)
        self.lob_box.add_widget(self.lob_label)
        self.lob_box.add_widget(self.lob_btns_container)
        self.lob_container.add_widget(self.lob_box)
        self.choose_list_of_blobs_popup = Popup(title = 'Choose validation trajectories',
                            content = self.lob_container,
                            size_hint = (.4, .4))

    def show_loading_text(self, *args):
        self.lob_label.text = "Loading..."

    def on_choose_list_of_blobs_btns_press(self, instance):
        if instance.text == 'With gaps':
            self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_path)
            self.list_of_blobs_save_path = CHOSEN_VIDEO.video.blobs_path
        else:
            self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_no_gaps_path)
            self.list_of_blobs_save_path = CHOSEN_VIDEO.video.blobs_no_gaps_path
        if not self.list_of_blobs.blobs_are_connected:
            self.list_of_blobs.reconnect()
        self.choose_list_of_blobs_popup.dismiss()
        self.populate_validation_tab()

    def populate_validation_tab(self):
        self.blobs_in_video = self.list_of_blobs.blobs_in_video
        self.count_scrollup = 0
        self.scale = 1
        self.loading_popup.dismiss()
        self.init_segmentZero()

    def get_first_frame(self):
        return CHOSEN_VIDEO.video.first_frame_first_global_fragment[CHOSEN_VIDEO.video.accumulation_trial]

    def do(self, *args):
        if CHOSEN_VIDEO.processes_to_restore is not None and CHOSEN_VIDEO.processes_to_restore['assignment']:
            CHOSEN_VIDEO.video.__dict__.update(CHOSEN_VIDEO.old_video.__dict__)
        if  CHOSEN_VIDEO.video.has_been_assigned and CHOSEN_VIDEO.video.has_crossings_solved:
            self.create_choose_list_of_blobs_popup()
            self.lob_btn1.bind(on_press = self.show_loading_text)
            self.lob_btn2.bind(on_press = self.show_loading_text)
            self.lob_btn1.bind(on_release = self.on_choose_list_of_blobs_btns_press)
            self.lob_btn2.bind(on_release = self.on_choose_list_of_blobs_btns_press)
            self.choose_list_of_blobs_popup.open()
        elif CHOSEN_VIDEO.video.has_been_assigned:
            self.loading_popup.open()
            self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_path)
            self.list_of_blobs_save_path = CHOSEN_VIDEO.video.blobs_path
            if not self.list_of_blobs.blobs_are_connected:
                self.list_of_blobs.reconnect()
            self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
            self.populate_validation_tab()
        else:
            self.warning_popup.open()

    def create_choose_individual_popup(self):
        self.choose_individual_container = BoxLayout(orientation = "vertical")
        self.choose_individual_label = CustomLabel(text = "Choose the identity of the individual to follow")
        self.choose_individual_input = TextInput(text ='', multiline=False)
        self.choose_individual_container.add_widget(self.choose_individual_label)
        self.choose_individual_container.add_widget(self.choose_individual_input)
        self.choose_individual_popup = Popup(title = 'Correcting identity',
                                            content = self.choose_individual_container,
                                            size_hint = (.4,.4))

    def on_enter_choose_individual(self, value):
        self.individual_to_follow = int(value.text)
        #init and bind keyboard again
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.choose_individual_popup.dismiss()

    def choose_individual(self, instance):
        if not hasattr(self, 'individual_to_follow'):
            self.create_choose_individual_popup()
            self.choose_individual_input.bind(on_text_validate = self.on_enter_choose_individual)
            self.choose_individual_popup.open()
        else:
            self.create_choose_individual_popup()
            self.choose_individual_label.text = "You already chose to validate individual " + str(self.individual_to_follow) + ". Type a different identity to change, or click outside the popup to continue."
            self.choose_individual_input.bind(on_text_validate = self.on_enter_choose_individual)
            self.choose_individual_popup.open()

    def init_segmentZero(self):
        self.add_widget(self.visualiser)
        self.colors = get_spaced_colors_util(CHOSEN_VIDEO.video.number_of_animals)
        self.button_box = BoxLayout(orientation='vertical', size_hint=(.3,1.))
        self.add_widget(self.button_box)
        self.choose_individual_btn = Button(text = "Choose individual to follow")
        self.choose_individual_btn.bind(on_press=self.choose_individual)
        self.button_box.add_widget(self.choose_individual_btn)
        self.next_cross_button = Button(text='Next crossing')
        self.next_cross_button.bind(on_press=self.go_to_next_crossing)
        self.button_box.add_widget(self.next_cross_button)
        self.previous_cross_button = Button(text='Previous crossing')
        self.previous_cross_button.bind(on_press=self.go_to_previous_crossing)
        self.button_box.add_widget(self.previous_cross_button)
        self.go_to_first_global_fragment_button = Button(text='First global fragment')
        self.go_to_first_global_fragment_button.bind(on_press = self.go_to_first_global_fragment)
        self.button_box.add_widget(self.go_to_first_global_fragment_button)
        self.save_groundtruth_btn = Button(text='Save updated identities')
        self.save_groundtruth_btn.bind(on_press=self.show_saving)
        self.save_groundtruth_btn.bind(on_release=self.save_groundtruth_list_of_blobs)
        self.save_groundtruth_btn.disabled = True
        self.button_box.add_widget(self.save_groundtruth_btn)
        self.compute_accuracy_button = Button(text = "Compute accuracy")
        self.compute_accuracy_button.disabled = False
        self.compute_accuracy_button.bind(on_press = self.compute_and_save_session_accuracy_wrt_groundtruth_APP)
        self.button_box.add_widget(self.compute_accuracy_button)
        self.visualiser.visualise_video(CHOSEN_VIDEO.video, func = self.writeIds, frame_index_to_start = self.get_first_frame())

    def go_to_crossing(self, direction = None, instance = None):
        if hasattr(self, 'individual_to_follow'):
            ind_is_present = True
            frame_index = int(self.visualiser.video_slider.value)

            while ind_is_present == True:
                if frame_index < CHOSEN_VIDEO.video.number_of_frames - 1 and frame_index > 0:
                    if direction == "next":
                        frame_index = frame_index + 1
                    elif direction == "previous":
                        frame_index = frame_index - 1
                    blobs_in_frame = self.blobs_in_video[frame_index]
                    ids_in_frame = set([blob.final_identity for blob in blobs_in_frame])
                    if not self.individual_to_follow in ids_in_frame:
                        ind_is_present = False
                        self.visualiser.video_slider.value = frame_index
                        self.visualiser.visualise(frame_index, func = self.writeIds)
                else:
                    break
        else:
            self.create_choose_individual_popup()
            self.choose_individual_input.bind(on_text_validate = self.on_enter_choose_individual)
            self.choose_individual_label.text = "Choose and individual first!"
            self.choose_individual_popup.open()

    def go_to_next_crossing(self,instance):
        self.go_to_crossing("next", instance)

    def go_to_previous_crossing(self,instance):
        self.go_to_crossing("previous", instance)

    def go_to_first_global_fragment(self, instance):
        self.visualiser.visualise(self.get_first_frame(), func = self.writeIds)
        self.visualiser.video_slider.value = self.get_first_frame()

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):

        if hasattr(self.visualiser, 'video_slider'):
            frame_index = int(self.visualiser.video_slider.value)
            if keycode[1] == 'left':
                frame_index -= 1
            elif keycode[1] == 'right':
                frame_index += 1
            elif keycode[1] == 'up':
                frame_index = self.go_to_crossing(direction = 'next')
            elif keycode[1] == 'down':
                frame_index = self.go_to_crossing(direction = 'previous')
            if frame_index is not None:
                self.visualiser.video_slider.value = frame_index
                self.visualiser.visualise(frame_index, func = self.writeIds)
        return True

    @staticmethod
    def get_clicked_blob(point, contours):
        """
        Get the contour that contains point
        """
        indices = [i for i, cnt in enumerate(contours) if cv2.pointPolygonTest(cnt, tuple(point), measureDist = False) >= 0]
        if len(indices) != 0:
            return indices[0]
        else:
            return None

    def apply_affine_transform_on_point(self, affine_transform_matrix, point):
        R = affine_transform_matrix[:,:-1]
        T = affine_transform_matrix[:,-1]
        return np.dot(R, np.squeeze(point)) + T

    def apply_inverse_affine_transform_on_point(self, affine_transform_matrix, point):
        inverse_affine_transform_matrix = cv2.invertAffineTransform(affine_transform_matrix)
        return self.apply_affine_transform_on_point(inverse_affine_transform_matrix, point)

    def apply_affine_transform_on_contour(self, affine_transform_matrix, contour):
        return np.expand_dims(np.asarray([self.apply_affine_transform_on_point(affine_transform_matrix, point) for point in contour]).astype(int),axis = 1)

    def get_blob_to_modify_and_mouse_coordinate(self):
        mouse_coords = self.touches[0]
        frame_index = int(self.visualiser.video_slider.value) #get the current frame from the slider
        blobs_in_frame = self.blobs_in_video[frame_index]
        contours = [getattr(blob, "contour") for blob in blobs_in_frame]
        if self.scale != 1:
            contours = [self.apply_affine_transform_on_contour(self.M, cnt) for cnt in contours]
        mouse_coords = self.fromShowFrameToTexture(mouse_coords)
        if self.scale != 1:
            mouse_coords = self.apply_inverse_affine_transform_on_point(self.M, mouse_coords)
        blob_ind = self.get_clicked_blob(mouse_coords, contours)
        if blob_ind is not None:
            blob_to_modify = blobs_in_frame[blob_ind]
            return blob_to_modify, mouse_coords
        else:
            return None, None

    def fromShowFrameToTexture(self, coords):
        """
        Maps coordinate in showFrame (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        if hasattr(CHOSEN_VIDEO.video, 'resolution_reduction') and  CHOSEN_VIDEO.video.resolution_reduction != 1:
            original_frame_width = int(CHOSEN_VIDEO.video.width * CHOSEN_VIDEO.video.resolution_reduction)
            original_frame_height = int(CHOSEN_VIDEO.video.height * CHOSEN_VIDEO.video.resolution_reduction)
        else:
            original_frame_width = int(CHOSEN_VIDEO.video.width)
            original_frame_height = int(CHOSEN_VIDEO.video.height)
        actual_frame_width, actual_frame_height = self.visualiser.display_layout.size
        self.offset = self.visualiser.footer.height
        coords[1] = coords[1] - self.offset
        wRatio = abs(original_frame_width / actual_frame_width)
        hRatio = abs(original_frame_height / actual_frame_height)
        ratios = np.asarray([wRatio, hRatio])
        coords =  np.multiply(coords, ratios)
        coords[1] = original_frame_height - coords[1]
        return coords

    @staticmethod
    def get_attributes_from_blobs_in_frame(blobs_in_frame, attributes_to_get):
        return {attr: [getattr(blob, attr) for blob in blobs_in_frame] for attr in attributes_to_get}

    def writeIds(self, frame):
        try:
            blobs_in_frame = self.blobs_in_video[int(self.visualiser.video_slider.value)]
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = self.visualiser.frame
            frame_number = blobs_in_frame[0].frame_number

            for blob in blobs_in_frame:
                cur_id = blob.final_identity
                cur_id_str = str(cur_id)
                roots = ['a-', 'd-', 'c-','i-', 'u-']
                if blob.user_generated_identity is not None:
                    root = roots[4]
                elif blob.identity_corrected_closing_gaps is not None and blob.is_an_individual:
                    root = roots[3]
                elif blob.identity_corrected_closing_gaps is not None:
                    root = roots[2]
                elif blob.identity_corrected_solving_duplication is not None:
                    root = roots[1]
                elif not blob.used_for_training:
                    root = roots[0]
                else:
                    root  = ''
                if isinstance(cur_id, int):
                    cur_id_str = root + cur_id_str
                    int_centroid = np.asarray(blob.centroid).astype('int')
                    cv2.circle(frame, tuple(int_centroid), 2, self.colors[cur_id], -1)
                    cv2.putText(frame, cur_id_str,tuple(int_centroid), font, 1, self.colors[cur_id], 3)
                    if blob.is_a_crossing or blob.identity_corrected_closing_gaps is not None or blob.assigned_identity == 0:
                        bounding_box = blob.bounding_box_in_frame_coordinates
                        if hasattr(blob, 'rect_color'):
                            rect_color = blob.rect_color
                        else:
                            rect_color = (255, 0, 0)
                        cv2.rectangle(frame, bounding_box[0], bounding_box[1], rect_color , 2)
                elif isinstance(cur_id, list):
                    for c_id, c_centroid in zip(cur_id, blob.interpolated_centroids):
                        c_id_str = root + str(c_id)
                        int_centroid = tuple([int(centroid_coordinate) for centroid_coordinate in c_centroid])
                        cv2.circle(frame, int_centroid, 2, self.colors[c_id], -1)
                        cv2.putText(frame, c_id_str, int_centroid, font, 1, self.colors[c_id], 3)

                    self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
                    self._keyboard.bind(on_key_down=self._on_keyboard_down)
                    if blob.is_a_crossing or blob.identity_corrected_closing_gaps is not None:
                        bounding_box = blob.bounding_box_in_frame_coordinates
                        if hasattr(blob, 'rect_color'):
                            rect_color = blob.rect_color
                        else:
                            rect_color = (255, 0, 0)
                        cv2.rectangle(frame, bounding_box[0], bounding_box[1], rect_color , 2)
                elif blob.assigned_identity is None:
                    bounding_box = blob.bounding_box_in_frame_coordinates
                    if hasattr(blob, 'rect_color'):
                        rect_color = blob.rect_color
                    else:
                        rect_color = (255, 0, 0)
                    cv2.rectangle(frame, bounding_box[0], bounding_box[1], rect_color , 2)
            if self.scale != 1:
                self.dst = cv2.warpAffine(frame, self.M, (frame.shape[1], frame.shape[0]))
                buf = cv2.flip(self.dst,0)
                buf = buf.tostring()
            else:
                buf = cv2.flip(frame,0)
                buf = buf.tostring()
            textureFrame = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.visualiser.display_layout.texture = textureFrame
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def propagate_groundtruth_identity_in_individual_fragment(self):
        modified_blob = self.blob_to_modify
        count_past_corrections = 1 #to take into account the modification already done in the current frame
        count_future_corrections = 0
        new_blob_identity = modified_blob.user_generated_identity
        if modified_blob.is_an_individual:
            print("is and individual")
            print("fragment_identifier ", modified_blob.fragment_identifier)
            current = modified_blob

            while len(current.next) == 1 and current.next[0].fragment_identifier == modified_blob.fragment_identifier:
                print("in first while")
                current.next[0]._user_generated_identity = current.user_generated_identity
                current = current.next[0]
                count_future_corrections += 1

            current = modified_blob

            while len(current.previous) == 1 and current.previous[0].fragment_identifier == modified_blob.fragment_identifier:
                print("in second while")
                current.previous[0]._user_generated_identity = current.user_generated_identity
                current = current.previous[0]
                count_past_corrections += 1

        #init and bind keyboard again
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def overwriteIdentity(self, instance):
        # enable buttons to save corrected version and compute the accuracy
        self.save_groundtruth_btn.disabled = False
        self.compute_accuracy_button.disabled = False
        if not self.blob_to_modify.is_a_crossing:
            self.blob_to_modify._user_generated_identity = self.individual_to_follow
            print("propagating identity")
            self.propagate_groundtruth_identity_in_individual_fragment()
            print("end of propagating identity")
            self.modify_id_popup.dismiss()
        self.visualiser.visualise(trackbar_value = int(self.visualiser.video_slider.value), func=self.writeIds)

    def on_press_show_saving(selg, *args):
        self.show_saving()

    def save_groundtruth_list_of_blobs(self, *args):
        self.go_and_save()
        self.popup_saving.dismiss()

    def go_and_save(self):
        self.list_of_blobs.save(CHOSEN_VIDEO.video, path_to_save = self.list_of_blobs_save_path)
        CHOSEN_VIDEO.video.save()

    def create_modify_identity_popup(self):
        self.container = BoxLayout(orientation = "vertical")
        self.id_to_modify = self.blob_to_modify.final_identity
        text = str(self.id_to_modify)
        self.old_id_box = BoxLayout(orientation="vertical")
        self.new_id_btn = Button(text = "Set identity to " + str(self.individual_to_follow))
        self.selected_label = CustomLabel(text='You selected animal:\n')
        self.selected_label_num = CustomLabel(text=text)
        self.container.add_widget(self.old_id_box)
        self.container.add_widget(self.new_id_btn)
        self.old_id_box.add_widget(self.selected_label)
        self.old_id_box.add_widget(self.selected_label_num)
        self.modify_id_popup = Popup(title='Correcting identity',
            content=self.container,
            size_hint=(.4,.4))
        self.modify_id_popup.color = (0.,0.,0.,0.)

    def modifyIdOpenPopup(self, blob_to_modify):
        if hasattr(self, 'individual_to_follow'):
            self.blob_to_modify = blob_to_modify
            self.create_modify_identity_popup()
            self.new_id_btn.bind(on_release = self.overwriteIdentity)
            self.modify_id_popup.open()
        else:
            self.create_choose_individual_popup()
            self.choose_individual_label.text = "Choose the identity you want to validate before modifying the identification. Type here:"
            self.choose_individual_input.bind(on_text_validate = self.on_enter_choose_individual)
            self.choose_individual_popup.open()

    def show_blob_attributes(self, blob_to_explore):
        self.container = BoxLayout()
        self.blob_to_explore = blob_to_explore
        self.show_attributes_box = BoxLayout(orientation="vertical")
        self.id_label = CustomLabel(text='Assigned identity: ' + str(blob_to_explore.final_identity))
        self.frag_id_label = CustomLabel(text='Fragment identifier: ' + str(blob_to_explore.fragment_identifier))
        self.accumulation_label = CustomLabel(text='Used for training: ' + str(blob_to_explore.used_for_training))
        self.in_a_fragment_label = CustomLabel(text='It is in an individual fragment: ' + str(blob_to_explore.is_in_a_fragment))
        self.fish_label = CustomLabel(text='It is a fish: ' + str(blob_to_explore.is_an_individual))
        self.container.add_widget(self.show_attributes_box)
        widget_list = [self.id_label, self.frag_id_label,
                        self.accumulation_label, self.in_a_fragment_label]
        [self.show_attributes_box.add_widget(w) for w in widget_list]
        self.blob_attr_popup = Popup(title='Blob attributes',
            content=self.container,
            size_hint=(.4,.4))
        self.blob_attr_popup.color = (0.,0.,0.,0.)
        self.blob_attr_popup.open()

    @staticmethod
    def get_index_of_fragment_identifier(fragment_identifier, blobs_in_frame):
        fragment_identifiers_in_frame = [blob.fragment_identifier for blob
                                        in blobs_in_frame]
        try:
            return fragment_identifiers_in_frame.index(fragment_identifier)
        except:
            return None

    def propagate_crossing_check_state(self):
        modified_blob = self.detected_blob_to_modify
        fragment_identifier = modified_blob.fragment_identifier
        blobs_in_video = self.list_of_blobs.blobs_in_video
        next_frame = modified_blob.frame_number + 1
        blob_index = self.get_index_of_fragment_identifier(fragment_identifier, blobs_in_video[next_frame])

        while blob_index is not None:
            blob = blobs_in_video[next_frame][blob_index]
            blob.rect_color = modified_blob.rect_color
            next_frame = next_frame + 1
            blob_index = self.get_index_of_fragment_identifier(fragment_identifier, blobs_in_video[next_frame])

        previous_frame = modified_blob.frame_number - 1
        blob_index = self.get_index_of_fragment_identifier(fragment_identifier, blobs_in_video[previous_frame])

        while blob_index is not None:
            blob = blobs_in_video[previous_frame][blob_index]
            blob.rect_color = modified_blob.rect_color
            previous_frame = previous_frame - 1
            blob_index = self.get_index_of_fragment_identifier(fragment_identifier, blobs_in_video[previous_frame])

        #init and bind keyboard again
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)


    def change_crossing_check_state(self, touch_type):
        if touch_type == 'left':
            self.detected_blob_to_modify.rect_color = (0, 255, 0)
        elif touch_type == 'right':
            self.detected_blob_to_modify.rect_color = (255, 0, 0)
        self.propagate_crossing_check_state()
        self.visualiser.visualise(trackbar_value = int(self.visualiser.video_slider.value), func = self.writeIds)

    def on_touch_down(self, touch):
        self.touches = []
        if self.visualiser.display_layout.collide_point(*touch.pos):
            if touch.button =='left':
                self.touches.append(touch.pos)
                self.detected_blob_to_modify, self.user_generated_centroids = self.get_blob_to_modify_and_mouse_coordinate()
                if self.detected_blob_to_modify is not None:
                    if  self.detected_blob_to_modify.is_an_individual:
                        self.modifyIdOpenPopup(self.detected_blob_to_modify)
                    else:
                        self.change_crossing_check_state(touch.button)
            elif touch.button == 'scrollup':
                self.count_scrollup += 1
                coords = self.fromShowFrameToTexture(touch.pos)
                rows, cols, channels = self.visualiser.frame.shape
                self.scale = 1.5 * self.count_scrollup
                self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
                self.dst = cv2.warpAffine(self.visualiser.frame,self.M,(cols,rows))
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='bgr')
                textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.visualiser.display_layout.texture = textureFrame
            elif touch.button == 'scrolldown':
                coords = self.fromShowFrameToTexture(touch.pos)
                rows,cols, channels = self.visualiser.frame.shape
                self.dst = self.visualiser.frame
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='bgr')
                textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.visualiser.display_layout.texture = textureFrame
                self.count_scrollup = 0
                self.scale = 1
            elif touch.button == 'right':
                self.touches.append(touch.pos)
                self.detected_blob_to_modify, self.user_generated_centroids = self.get_blob_to_modify_and_mouse_coordinate()
                if self.detected_blob_to_modify is not None:
                    if  self.detected_blob_to_modify.is_an_individual:
                        self.show_blob_attributes(self.detected_blob_to_modify)
                    else:
                        self.change_crossing_check_state(touch.button)
        else:
            self.scale = 1
            self.disable_touch_down_outside_collided_widget(touch)

    def disable_touch_down_outside_collided_widget(self, touch):
        return super(IndividualValidator, self).on_touch_down(touch)

    def get_groundtruth_path(self):
        groundtruth_path = os.path.join(CHOSEN_VIDEO.video.video_folder, '_individual_' + str(self.individual_to_follow) + '_groundtruth.npy')
        return groundtruth_path if os.path.isfile(groundtruth_path) else None

    def on_groundtruth_popup_button_press(self, instance):
        if instance.text == "Use pre-existent ground truth":
            self.groundtruth = np.load(self.groundtruth_path).item()
            self.plot_groundtruth_statistics()
            self.popup_start_end_groundtruth.dismiss()
        else:
            self.gt_start_end_container.remove_widget(self.gt_start_end_btn1)
            self.gt_start_end_container.remove_widget(self.gt_start_end_btn2)
            self.gt_start_end_label.text = "Insert the start and ending frame (e.g. 100 - 2050) on which the ground truth has been computed"
            self.gt_start_end_text_input = TextInput(text ='', multiline=False)
            self.gt_start_end_container.add_widget(self.gt_start_end_text_input)
            self.gt_start_end_text_input.bind(on_text_validate = self.on_enter_start_end)

    def create_frame_interval_popup(self):
        if not hasattr(self, 'individual_to_follow'):
            self.create_choose_individual_popup()
            self.choose_individual_label.text = "Choose the identity you want to validate before trying to compute an individual accuracy:"
            self.choose_individual_input.bind(on_text_validate = self.on_enter_choose_individual)
            self.choose_individual_popup.open()
            return True
        else:
            self.gt_start_end_container = BoxLayout(orientation = "vertical")
            self.groundtruth_path = self.get_groundtruth_path()
            if self.groundtruth_path is not None:
                if self.save_groundtruth_btn.disabled:
                    self.groundtruth = np.load(self.groundtruth_path).item()
                    self.plot_groundtruth_statistics()
                    return True
                if not self.save_groundtruth_btn.disabled:
                    self.gt_start_end_label = CustomLabel(text = "A pre-existent ground truth file has been detected. Do you want to use it to compute the accuracy or use a new one?")
                    self.gt_start_end_btn1 = Button(text = "Use pre-existent ground truth")
                    self.gt_start_end_btn2 = Button(text = "Generate new ground truth")
                    self.gt_start_end_container.add_widget(self.gt_start_end_label)
                    self.gt_start_end_container.add_widget(self.gt_start_end_btn1)
                    self.gt_start_end_container.add_widget(self.gt_start_end_btn2)
                    self.gt_start_end_btn1.bind(on_press = self.on_groundtruth_popup_button_press)
                    self.gt_start_end_btn2.bind(on_press = self.on_groundtruth_popup_button_press)
            else:
                if self.save_groundtruth_btn.disabled:
                    self.gt_start_end_label = CustomLabel(text = "No pre-existent groundtruth file has been detected. Validate the video to compute a ground truth first.\n\n Need help? To modify a wrong identity click on the badly identified animal and fill the popup. Use the mouse wheel to zoom if necessary.")
                    self.gt_start_end_container.add_widget(self.gt_start_end_label)
                else:
                    self.gt_start_end_label = CustomLabel(text = "Insert the start and ending frame (e.g. 100 - 2050) on which the ground truth has been computed")
                    self.gt_start_end_container.add_widget(self.gt_start_end_label)
                    self.gt_start_end_text_input = TextInput(text ='', multiline=False)
                    self.gt_start_end_container.add_widget(self.gt_start_end_text_input)
                    self.gt_start_end_text_input.bind(on_text_validate = self.on_enter_start_end)

            self.popup_start_end_groundtruth = Popup(title='Groundtruth Accuracy - Frame Interval',
                        content=self.gt_start_end_container,
                        size_hint=(.4,.4))

    def on_enter_start_end(self, value):
        start, end = self.gt_start_end_text_input.text.split('-')
        self.gt_start_frame = int(start)
        self.gt_end_frame = int(end)
        self.generate_groundtruth()
        self.save_groundtruth()
        self.plot_groundtruth_statistics()
        if not self.prevent_open_popup:
            self.popup_start_end_groundtruth.dismiss()

    def generate_groundtruth(self):
        self.groundtruth = generate_individual_groundtruth(CHOSEN_VIDEO.video,
                                                    blobs_in_video = self.blobs_in_video,
                                                    start = self.gt_start_frame,
                                                    end = self.gt_end_frame,
                                                    validated_identity = self.individual_to_follow,
                                                    save_gt = False)

    def save_groundtruth(self):
        self.groundtruth.save()

    def plot_groundtruth_statistics(self):
        individual_blobs_in_video_groundtruth = [blob for blob in self.groundtruth.individual_blobs_in_video
                                        if (blob.frame_number >= self.groundtruth.start
                                        and blob.frame_number <= self.groundtruth.end)]
        blobs_in_video = self.blobs_in_video[self.groundtruth.start:self.groundtruth.end]
        comparison_info = get_individual_accuracy_wrt_groundtruth(CHOSEN_VIDEO.video, individual_blobs_in_video_groundtruth)
        comparison_info['number_of_occluded_frames'] = self.groundtruth.end - self.groundtruth.start - len(individual_blobs_in_video_groundtruth)
        pprint(comparison_info)
        self.frames_with_errors = comparison_info['frames_with_errors']
        self.mistaken_identities = comparison_info['mistaken_identities']
        self.accuracy = comparison_info['accuracy']
        self.plot_final_statistics()
        self.statistics_popup.open()
        comparison_info['start-end'] = (self.groundtruth.start, self.groundtruth.end)
        if not hasattr(CHOSEN_VIDEO.video, 'individual_groundtruths'):
            CHOSEN_VIDEO.video.individual_groundtruths = [comparison_info]
        else:
            CHOSEN_VIDEO.video.individual_groundtruths.append(comparison_info)
        CHOSEN_VIDEO.video.save()

    def compute_and_save_session_accuracy_wrt_groundtruth_APP(self, *args):
        self.prevent_open_popup = self.create_frame_interval_popup()
        if not self.prevent_open_popup:
            self.popup_start_end_groundtruth.open()

    def plot_final_statistics(self):
        content = BoxLayout(orientation = "vertical")
        self.statistics_popup = Popup(title = "Statistics",
                                    content = content,
                                    size_hint = (.5, .5))
        frames_with_errors_lbl = CustomLabel(text = "Frames with errors: " + str(self.frames_with_errors))
        mistaken_identities_lbl = CustomLabel(text = "Mistaken identities: " + str(self.mistaken_identities))
        accuracy_lbl = CustomLabel(text = "Accuracy: " + str(self.accuracy))
        content.add_widget(frames_with_errors_lbl)
        content.add_widget(mistaken_identities_lbl)
        content.add_widget(accuracy_lbl)
