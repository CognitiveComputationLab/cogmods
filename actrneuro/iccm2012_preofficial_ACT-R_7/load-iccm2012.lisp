;;; File : load-iccm2012.lisp

(load "globalvars.lisp")
(load "utilities.lisp")

(setf *currentswitches* *iccm-currentswitches*)
(setf *experiment* "iccm")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; experiment code                                                    ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun experiment-key-press (model key)
    (setf *total-time* (get-time))
    (setf *response* (string key)))


(defun do-experiment (model &key (run -1) (vpset -1) (ans -1) (rt -1) (lf -1) (bold-scale -1) (neg-bold-scale -1) (bold-exp -1) (neg-bold-exp -1) (bold-positive -1) (bold-negative -1) (verbose nil) (run-amount nil) )
    (cond
        ((string-equal *model* "pmm") (setf *strategy* 'pmm))
        ((string-equal *model* "any") (setf *strategy* 'any)))
    (when (not (check-arguments model))
        (return-from do-experiment nil))
    (when (null *model-loaded*)
        (load "latest9.lisp")
        (setf *model-loaded* t))


    ;; HACK FOR KEYBOARD TO MAKE SURE ACT-R 7 AND ACT-R 6 HAVE THE SAME LOCATION FOR SPACE BAR 
    (specify-key *act-r-virtual-keyboard* 6 6 "space")

    (reset)
    (sgp-fct `(:ans ,ans
                          :rt ,rt
                          :lf ,lf
                          :v ,verbose
                          :bold-scale ,bold-scale
                          :neg-bold-scale ,neg-bold-scale
                          :bold-exp ,bold-exp
                          :neg-bold-exp ,neg-bold-exp
                          :bold-positive ,bold-positive
                          :bold-negative ,bold-negative
                          ))
    (setf *trial* 0)
    (setf *model-results* nil)

    (choose-vpset vpset) ; write respective tasks to *taskarray* depending on vpset

    (dolist (task *taskarray*)

        (setf *response* nil)
        (setf (focus-position *modelfocus*) 1)
        (setf (focus-position *premisefocus*) 1)
        (setf (focus-position *conclusionfocus*) 1)

        (setf *premises* nil)
        (setf *trial* (+ 1 *trial*))

        (let*
            ((nop          (switches-numberofpremises *currentswitches*))
            (premises      (dotimes (premisenumber nop *premises*)
                (setf *premises* (append *premises* (cons (cons (+ 1 premisenumber) (nth premisenumber task)) nil)))))
            (conclusion    (nth (switches-numberofpremises *currentswitches*) task))
            (model         (nth (+ (switches-numberofpremises *currentswitches*) 1) task))
            (taskid        (nth (+ (switches-numberofpremises *currentswitches*) 2) task))
            (expected1     (nth (+ (switches-numberofpremises *currentswitches*) 3) task))
            (determinacy   (nth (+ (switches-numberofpremises *currentswitches*) 4) task))
            (expected2     (nth (+ (switches-numberofpremises *currentswitches*) 5) task))
            (window        (open-exp-window "" :width 300 :height 300 :x 0 :y 0 :visible *window-visible*)))

            (when verbose
                (format t "Presenting vpset-taskid ~S-~S~%" vpset taskid))
            (install-device window)



            (add-act-r-command "experiment-key-press" 'experiment-key-press "Experiment task output-key monitor")
            (monitor-act-r-command "output-key" "experiment-key-press")



            (setf *trial-start-time* (get-time))



            ;; BOLD
            (record-history "buffer-trace" "goal" "manual" "visual" "imaginal" "aural" "retrieval")
            (setf *bold-response-start* (mp-time))
            (let
                ((inc (no-output (car (sgp :bold-inc)))))
                (setf *bold-response-start* (* inc (floor *bold-response-start* inc))))




            ;; BEFORE THE PRESENTATION OF THE PREMISES BEGINS, ONE HAS TO PRESS THE KEY " " (space) FOR "Press space to start"
            (setf *response* nil)

            (when verbose

                (format t "***************************************************~%")
                (format t "***************************************************~%")
                (format t "***************** initial S print *****************~%")
                (format t "***************************************************~%")
                (format t "***************************************************~%")
            )
            (add-text-to-exp-window window "S" :x 140 :y 200)
            (when verbose
                (format t "Press space to start~%"))
            (run-until-condition 'answer-given )


            (clear-exp-window)

            ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present premises ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
            (present-premises premises verbose window)

            ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present the conclusion ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
            (present-conclusion conclusion verbose window)

            ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present model ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
            (present-model model verbose window)



            (remove-act-r-command-monitor "output-key" "experiment-key-press")
            (remove-act-r-command "experiment-key-press")



            (setf *bold-response-end* (mp-time))




            ;; BOLD END


            (bold-response-to-file run vpset taskid verbose *bold-response-start* *bold-response-end*)

            ;; Write measured data into lists
            ;; In the experiment the participants had to answer with Backarrow and Forwardarrow, but for evaluating the log files we use the terms C and N,
            ;; so the responses are replaced by these terms.
            (when (null *response*)
                (setf *response* "NA"))

                (when (string-equal *firstresponse* "left-arrow")
                    (setf *firstresponse* "c"))

                (when (string-equal *firstresponse* "right-arrow")
                    (setf *firstresponse* "n"))

                (when (string-equal *response* "left-arrow")
                    (setf *response* "c"))

                (when (string-equal *response* "right-arrow")
                    (setf *response* "n"))

            (let
                ((rt1 (- *first-total-time* *start-time*))  ;; conclusion time
                (rt2 (- *total-time* *start-time-model*))   ;; model time
                (trial *trial*)
                (response1 *firstresponse*)                 ;; conclusion response
                (response2 *response*))                     ;; model response


				; In the task files a correct expected answer is encoded by 1 and an incorrect expected answer is encoded by 0.
				; For comparing the pressed key by the model the information which key represents which answer the switches (defined in the globalvars) are used.

                ; conclusion
                    (cond
                        ((or (and (string-equal response1 (switches-correct-term *currentswitches*)) (equal expected1 1))
                             (and (string-equal response1 (switches-not-correct-term *currentswitches*)) (equal expected1 0)))
                            (setf *correct1* 1))
                        (t (setf *correct1* 0)))

                ; model
                (cond
                    ((or (and (string-equal response2 (switches-correct-term *currentswitches*)) (equal expected2 1))
                         (and (string-equal response2 (switches-not-correct-term *currentswitches*)) (equal expected2 0)))
                        (setf *correct2* 1))
                    (t (setf *correct2* 0)))

                (cond ((string-equal response1 "NA")
                    (setq rt1 "NA")))
                (cond ((string-equal response2 "NA")
                    (setq rt2 "NA")))

                (unless (string-equal response1 " ")
                    (setf *model-results*
                        (append *model-results*
                            (list
                                run
                                vpset
                                trial
                                taskid
                                rt1
                                *correct1*
                                response1
                                expected1
                                rt2
                                *correct2*
                                response2
                                expected2
                                determinacy
                                ans
                                rt
                                lf
                                ))))
                ; When the model fails to answer two times consecutively, the model is resetted, i.e. the goal buffer and the declarative memory
                (when (string-equal response2 "NA")
					(format t "NA found.")
					(when (string-equal *previous-response* "NA")
						(format t "Model resetted.")
						(reset)
						(no-output (sgp-fct `(:ans ,ans :rt ,rt :lf ,lf :v ,verbose)))
						(setf *response* nil))
				)
				(setf *previous-response* *response*)
            ))
    )


    )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present the premises ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun present-premises (premises verbose window)
    (dolist (premise premises)
        (let
            ((term1 (nth 1 premise))
            (modus (nth 2 premise))
            (term2 (nth 3 premise)))

            ;;;;;;;;;;;;;;; 1.5 seconds: present first term of the premise ;;;;;;;;;;;;;;;
            (setf *response* nil)
            (if (string-equal modus "l")
                (add-text-to-exp-window window term1 :x 90 :y 140)
                (add-text-to-exp-window window term1 :x 190 :y 140))
            (when verbose
                (format t "***************************************************~%")
                (format t "***************************************************~%")
                (format t "Presenting premise, term 1: ~S with ~S~%" term1 modus)
                (format t "***************************************************~%")
                (format t "***************************************************~%")
            )

            (run-full-time 1.5)

            (clear-exp-window)

            ;;;;;;;;;;;;;;; 1.5 seconds: present second term of the premise ;;;;;;;;;;;;;;;
            (if (string-equal modus "l")
                (add-text-to-exp-window window term2 :x 190 :y 140)
                (add-text-to-exp-window window term2 :x 90 :y 140))
            (when verbose
                (format t "***************************************************~%")
                (format t "***************************************************~%")

                (format t "Presenting premise, term 2: ~S with ~S~%" term2 modus)
                (format t "***************************************************~%")
                (format t "***************************************************~%")
            )

            (run-full-time 1.5)
            (clear-exp-window)




            (add-text-to-exp-window window "S" :x 140 :y 200)

            (when verbose

            (format t "***************************************************~%")
            (format t "***************************************************~%")
            (format t "**********   print S ******************************~%")
            (format t "***************************************************~%")
            (format t "***************************************************~%")
            )
            (run-until-condition 'answer-given)

            (clear-exp-window)
            ;; if no key was pressed, get the current time
            (unless *response*
                (setf *total-time* (get-time))
                (setf *response* "NA"))
            ))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present the conclusion ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun present-conclusion (conclusion verbose window)

    (setf *response* nil)
    (setf *start-time* (get-time))

    ;; PRESENT CONCLUSION
    (let
        ((term1 (nth 0 conclusion))
        (modus (nth 1 conclusion))
        (term2 (nth 2 conclusion)))

        ;;;;;;;;;;;;;;; 1.5 seconds: present first term of conclusion ;;;;;;;;;;;;;;;
        (setf *response* nil)
        (if (string-equal modus "l")
            (add-text-to-exp-window window term1 :x 90 :y 140)
            (add-text-to-exp-window window term1 :x 190 :y 140))
        (when verbose
                (format t "***************************************************~%")
                (format t "***************************************************~%")
                (format t "Presenting conclusion, term 1: ~S with ~S~%" term1 modus)
                (format t "***************************************************~%")
                (format t "***************************************************~%")
            )
        (run-full-time 1.5)
        (clear-exp-window)
        ;;;;;;;;;;;;;;; 1.5 seconds: present second term of conclusion ;;;;;;;;;;;;;;;
        (when verbose
                (format t "***************************************************~%")
                (format t "***************************************************~%")
                (format t "Presenting conclusion, term 1: ~S with ~S~%" term2 modus)
                (format t "***************************************************~%")
                (format t "***************************************************~%")
            )
        (if (string-equal modus "l")
            (add-text-to-exp-window window term2 :x 190 :y 140)
            (add-text-to-exp-window window term2 :x 90 :y 140))
        (run-full-time 1.5)


        (clear-exp-window)
        (setf *response* nil)
        (add-text-to-exp-window window "A" :x 140 :y 200)    ; "Signals the model that it is allowed to answer now"
        (run 12)    ; The model has up to 12 seconds to answer
        (clear-exp-window)
        ;; if no key was pressed, get the current time
        (unless *response*
            (setf *total-time* (get-time))
            (setf *response* "NA"))
        (setf *firstresponse* *response*)
        (setf *first-total-time* *total-time*)
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present the model ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun present-model (model verbose window)
    (run-full-time 1)
    (let
        ((curpos 120))

        (dotimes (count 4)
            (add-text-to-exp-window window (nth count model) :x curpos :y 140)
            (setf curpos (+ curpos 20))))
    (when verbose
        (format t "Presenting model~%"))
    ;; Answer procedure for model
    (setf *response* nil)
    (setf *start-time-model* (get-time))
    (add-text-to-exp-window window "A" :x 140 :y 200)    ; "Signals the model that it is allowed to answer now"
    (run 12)    ; The model has up to 12 seconds to answer
    (clear-exp-window)

    ;; if no key was pressed, get the current time
    (unless *response*
        (setf *total-time* (get-time))
        (setf *response* "NA")))


(defun trace-to-file (times model &key (filename "trace.txt"))
    (when (not (check-arguments model))
        (return-from trace-to-file nil))
    (with-open-file (*standard-output* filename
        :direction :output
        :if-exists :supersede
        :if-does-not-exist :create)
        (do-n times model "00000" :verbose t))
    (format t "~%Done."))



(defun bold-response-to-file (run vp taskid verbose start end)
"This function writes the predicted BOLD responses for a specific trial to a file.
It is called from the method  RPM-WINDOW-KEY-EVENT-HANDLER that returns the key that
has been pressed by the model.  The endtime is taken in the same method.The starttime
is taken in the first production READ-MODE that is the first production to fire."

  (let ((tnum nil)
        (sid nil)
        (run (write-to-string run))
        (vp (write-to-string vp))
        (taskid (write-to-string taskid))

        )

        ;(setf bolds    (predict-bold-response start end t))
        ;(print-bold-table bolds)




        (ensure-directories-exist  (concatenate 'string "/home/rob/Desktop/actrneuro/iccm2012_preofficial_ACT-R_7/log/bolds/run-" run "/"))
        (ensure-directories-exist  (concatenate 'string "/home/rob/Desktop/actrneuro/iccm2012_preofficial_ACT-R_7/log/bolds/run-" run "/" vp "/"))
        (ensure-directories-exist  (concatenate 'string "/home/rob/Desktop/actrneuro/iccm2012_preofficial_ACT-R_7/log/bolds/run-" run "/" vp "/" taskid "/"))

        (no-output (sgp-fct `(:v , t)))
        (with-open-file (*standard-output*
            (concatenate 'string  "/home/rob/Desktop/actrneuro/iccm2012_preofficial_ACT-R_7/log/bolds/run-" run "/" vp "/" taskid "/bold-response.dat")
                :direction :output
                :if-exists :supersede
                :if-does-not-exist :create)
                (print-bold-table (predict-bold-response start end nil))
        )

        (no-output (sgp-fct `(:v , verbose)))


    )

)

