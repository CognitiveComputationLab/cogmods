;; This file contains the ACT-R Screen simulation. This file presents visual stimuli to a loaded model and processes the feedback from the model

;; load variables and helpers
(load "globalvars.lisp")
(load "utilities.lisp")


(defun experiment-key-press (model key)
    "This function will be triggerd when the ACT-R models presses a key"
    (ignore model)
    (setf *total-time* (get-time))
    (setf *response* (string key)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; experiment code                                                    ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun do-experiment (model &key (run -1) (vpset -1) (ans -1) (rt -1) (lf -1) (bold-scale -1) (neg-bold-scale -1) (bold-exp -1) (neg-bold-exp -1) (bold-positive -1) (bold-negative -1) (verbose t) (run-amount nil) )
    (cond
        ((string-equal *model* "pmm") (setf *strategy* 'pmm))
        ((string-equal *model* "any") (setf *strategy* 'any)))

    (when (null *model-loaded*)
        (print "MODELMODELMDOEL")
        (load "latest9.lisp")
        (setf *model-loaded* t))


    ;; HACK FOR KEYBOARD TO MAKE SURE ACT-R 7 AND ACT-R 6 HAVE THE SAME LOCATION FOR SPACE BAR this slighty affects BOLD prediction for Manual Bugger and response Time
    (no-output (specify-key *act-r-virtual-keyboard* 6 6 "space"))

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
            ((nop          2)
            (premises      (dotimes (premisenumber nop *premises*)
                (setf *premises* (append *premises* (cons (cons (+ 1 premisenumber) (nth premisenumber task)) nil)))))
            (conclusion      (nth 2 task))
            (taskid             (nth 3 task))
            (expected         (nth 4 task))

            (window        (open-exp-window "" :width 300 :height 300 :x 0 :y 0 :visible *window-visible*)))



            (when verbose
                (format t "Presenting run-vpset-taskid ~S-~S-~S~%" run vpset taskid))
            (install-device window)



            (add-act-r-command "experiment-key-press" 'experiment-key-press "Experiment task output-key monitor")
            (monitor-act-r-command "output-key" "experiment-key-press")



            (setf *trial-start-time* (get-time))



            ;; BOLD
            ;; (record-history "buffer-trace" "goal" "manual" "visual" "imaginal" "retrieval" "production")
            (record-history "buffer-trace" "visual" "visual-location" "aural" "aural-location")
			;; (record-history "buffer-trace" "goal")


            ;; BEFORE THE PRESENTATION OF THE PREMISES BEGINS, ONE HAS TO PRESS THE KEY " " (space) FOR "Press space to start"
            (setf *response* nil)
            (when verbose

                (format t "***************************************************~%")
                (format t "***************************************************~%")
                (format t "***************** initial S print *****************~%")
                (format t "***************************************************~%")
                (format t "***************************************************~%")
            )

            ;; fang 2006
            ;; (add-text-to-exp-window window "S" :x 140 :y 200)
            ;; fang 2009
            ;; (new-other-sound 'schliessen 1 0 1 (mp-time) 'external 'word) 
            ;; (new-other-sound 'schliessen .5 .2 .15 (mp-time) 'external 'other nil '(:both value t) '(:evt volume large))

            (new-other-sound "schliessen" 1 0 0.25 (mp-time) 'external 'word) 








            (when verbose
                (format t "Press space to start~%"))

            (run-full-time 1)
            ;; (run-until-condition 'answer-given )


            

            (clear-exp-window)

            (run-full-time 1)





            (setf *bold-response-start* (mp-time))
            (let
                ((inc (no-output (car (sgp :bold-inc)))))
                (setf *bold-response-start* (* inc (floor *bold-response-start* inc))))



            ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present premises ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
            (present-premises premises verbose)


            (setf *response* nil)

            ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present the conclusion ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
            (present-conclusion conclusion verbose)


            ;; if no key was pressed, get the current time
            (unless *response*
                (setf *total-time* (get-time))
                (setf *response* "NA"))
            (setf *firstresponse* *response*)
            (setf *first-total-time* *total-time*)


            (remove-act-r-command-monitor "output-key" "experiment-key-press")
            (remove-act-r-command "experiment-key-press")



            ;; BOLD END


            (bold-response-to-file run vpset taskid verbose *bold-response-start* (+ *bold-response-end* 5))

            ;; Write measured data into lists
            ;; In the experiment the participants had to answer with Backarrow and Forwardarrow, but for evaluating the log files we use the terms C and N,
            ;; so the responses are replaced by these terms.



            (let*
                ((retrievaltime (- *total-time* *start-time*))

                (response *response*)
                (correct (if (or (and (string-equal response "r") (equal expected 1)) (and (string-equal response "f") (equal expected 0))) 1 0)))

                (if (equal expected 1)
                    (setf expected "r")
                    (setf expected "f")
                )

                (unless (string-equal response " ")
                    (setf *model-results*
                        (append
                            *model-results*
                            (list
                                run
                                *trial*
                                taskid
                                retrievaltime
                                correct      ; 0 if response != expected, 1 otherwise
                                response     ; key pressed
                                expected
                                model
                                ans
                                rt
                                lf
                                bold-scale
                                neg-bold-scale
                                bold-exp
                                neg-bold-exp
                                bold-positive
                                bold-negative
                                run-amount

                                )))))))
     *model-results*)











;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present the premises ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun present-premises (premises verbose)
    (dolist (premise premises)
        (let
            ((term1 (nth 1 premise))
            (modus (nth 2 premise))
            (term2 (nth 3 premise)))

            ;;;;;;;;;;;;;;; 1.5 seconds: present first term of the premise ;;;;;;;;;;;;;;;
            (setf *response* nil)
            (if (string-equal modus "l")
                ;; fang 2006
                ;; (add-text-to-exp-window window term1 :x 90 :y 140)
                ;; (add-text-to-exp-window window term1 :x 190 :y 140)
                ;; fang 2009
                (new-word-sound term1 (mp-time) 'external-left)       
                (new-word-sound term1 (mp-time) 'external-right)  
            )
                            (print-audicon)        

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
                ;; fang 2006
                ;; (add-text-to-exp-window window term2 :x 190 :y 140)
                ;; (add-text-to-exp-window window term2 :x 90 :y 140)
                ;; fang 2009
                (new-word-sound term2 (mp-time) 'external-right)       
                (new-word-sound term2 (mp-time) 'external-left)    
  
            )
            (print-audicon)        

            (when verbose
                (format t "***************************************************~%")
                (format t "***************************************************~%")

                (format t "Presenting premise, term 2: ~S with ~S~%" term2 modus)
                (format t "***************************************************~%")
                (format t "***************************************************~%")
            )

            (run-full-time 1.5)
            (clear-exp-window)



            (clear-exp-window)
            ;; if no key was pressed, get the current time


            (run-full-time 1)

            ))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Present the conclusion ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun present-conclusion (conclusion verbose)

    (setf *response* nil)
    (setf *start-time* (get-time))
    (when verbose

        (print "****************************************")
        (print "****************************************")
        (print "****************************************")
        (print "********   in conclusions  *************")
        (print "****************************************")
    )
    ;; PRESENT CONCLUSION
    (let
        ((term1 (nth 0 conclusion))
        (modus (nth 1 conclusion))
        (term2 (nth 2 conclusion)))

        (setf *response* nil)
        (if (string-equal modus "l")
            ;; fang 2006
            ;; (add-text-to-exp-window window term1 :x 90 :y 140)
            ;; (add-text-to-exp-window window term1 :x 190 :y 140)
            ;; fang 2009
            (new-word-sound term1 (mp-time) 'external-left)       
            (new-word-sound term1 (mp-time) 'external-right)  
        )
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
            ;; fang 2006
            ;; (add-text-to-exp-window window term2 :x 190 :y 140)
            ;; (add-text-to-exp-window window term2 :x 90 :y 140)
            ;; fang 2009
            (new-word-sound term2 (mp-time) 'external-right)       
            (new-word-sound term2 (mp-time) 'external-left)  
        )
        (run-full-time 1.5)

        (run-full-time 1)

        (setf *bold-response-end* (mp-time))

        (clear-exp-window)


        (run-full-time 11)    ; The model has up to 12 seconds to answer

    )
)