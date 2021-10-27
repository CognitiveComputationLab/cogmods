;; This file contains public variables and helper functions for the model and the environment

;; Public variables shared by the environment and the model
(defvar *bold-response-start*)
(defvar *bold-response-end*)

;; Public variables used by utilities.lisp
(defvar new-bold)
(defvar length)
(defvar formatstring)
(defvar formatcounter)
(defvar x-list)
(defvar *vp-count* 1)


;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Model helper functions
(defun move-focus-right (focus)
  "Moves focus on slot to the right"
  (let* ((pos (+ (focus-position focus) 1)))
    (setf (focus-position focus) pos)
    (append  (list (read-from-string
            (concatenate 'string "pos"
                 (write-to-string (focus-position focus))))))))


(defun move-focus-left (focus)
  "Moves focus on slot to the left"
  (let* ((pos (- (focus-position focus) 1)))
     (setf (focus-position focus) pos)
    (append  (list (read-from-string
            (concatenate 'string "pos"
                 (write-to-string (focus-position focus))))))))


(defun set-focus (pos focus)
  "Moves focus to specific position in the mental model regarding the argument 'pos'"
  (let* ((posname (symbol-name pos))
     (num (parse-integer
           (subseq posname 3 (length posname)))))
    (setf (focus-position focus) num)))


(defun extend-left (leftborder)
  "This function returns a new slot name for mental model chunk. It decrements the number of
  the slot representing the far left content. i.e. POS1  ->   POS0"
  (let* ((posname (symbol-name leftborder))
     (newslot (concatenate 'string "pos"
                   (write-to-string (- (parse-integer (subseq posname 3 (length posname))) 1)))))
    (read-from-string newslot)))


(defun extend-right (rightborder)
  "This function returns a new slot name for mental model chunk. It increments the number of
  the slot representing the far right content. i.e. POS1  ->   POS2"
  (let* ((posname (symbol-name rightborder))
    (newslot (concatenate 'string "pos"
                (write-to-string (+ (parse-integer (subseq posname 3 (length posname))) 1)))))
    (read-from-string newslot)))


;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Environment helper functions

(defun do-n (&optional (n nil) (model  "any") (ans nil) (rt nil) (lf nil) (bold-scale nil) (neg-bold-scale nil) (bold-exp nil) (neg-bold-exp nil) (bold-positive nil) (bold-negative nil) (verbose t))
    "Writes the model results of n runs to a tab delimited file. This is the entry point if multiple runs are required.
    n               =   amount of runs
    model           =   pmm or any {pmm for preferred mental model behavior}
    rt              =   :rt parameter of ACT-R
    lf              =   :lf parameter of ACT-R
    bold-scale      =   :bold-scale parameter of ACT-R
    neg-bold-scale  =   :neg-bold-scale of ACT-R
    bold-exp        =   :bold-exp parameter of ACT-R
    neg-bold-exp    =   :neg-bold-exp parameter of ACT-R
    bold-positive   =   :bold-positive parameter of ACT-R
    bold-negative   =   :bold-negative parameter of ACT-R
    verbose         =   enable/disable model output
    "

    ;; input processing
    (when (not n)
      (setf n 1)
      (print-warning "n (run amount) has not been set and will be on default: 1")
    )

    (when (not ans)
      (setf ans nil)
      (print-warning "ans (activation noise) has not been set and will be on default: nil")
    )

    (when (not rt)
      (setf rt 0)
      (print-warning "rt (retrieval threshold) has not been set and will be on default: 0")
    )

    (when (not lf)
      (setf lf 1)
      (print-warning "lf (latency factor) has not been set and will be on default: 1")
    )

    (when (not bold-scale)
      (setf bold-scale 1)
      (print-warning "bold-scale has not been set and will be on default: 1")
    )

    (when (not neg-bold-scale)
      (setf neg-bold-scale 1)
      (print-warning "neg-bold-scale has not been set and will be on default: 1")
    )


    ;; ensure model is 'pmm' or 'any'
    (when (not (check-arguments model))
      (return-from do-n nil))

    ;; iterate trough all persons
    (do ((vpset *vp-count* (+ vpset 1)))
        ((> vpset *vp-count*))
        (setf *runs* nil)
        ;; run n times for each person
        (dotimes (count n)
            (let* ((run (+ 1 count))
                  ;; run a complete experiment
                  (result (do-experiment model :run run :vpset vpset :ans ans :rt rt :lf lf
                                            :bold-scale bold-scale :neg-bold-scale neg-bold-scale :bold-exp bold-exp
                                            :neg-bold-exp neg-bold-exp :bold-positive bold-positive :bold-negative bold-negative
                                             :verbose verbose :run-amount n)))
        ;; safe results
        (setf *runs* (append *runs* result))
        (format t "~&------------------------------> RUN ~d: VPSET ~d: ANS: ~d RT: ~d LF: ~d <------------------------------~%" run vpset ans rt lf)))
        (let ((filename (concatenate 'string  "./log/model/fang6.dat")))

      (with-open-file (*standard-output* filename
                         :direction :output
                         :if-exists :supersede
                         :if-does-not-exist :create)
        (print-table *runs*))

      (format t "~%File ~s written.~%" filename)))
    (format t "~%")
)




(defun print-table (runs)
  "This function formats a list of runs to multiple strings to print them in a table."
  (format t "~&RUN~15TTRIAL~30TSID~45TRtime~60TCORRECT~75TRESPONSE~90TEXPECTED~105TMODEL~120TANS~135TRT~150TLF~165TBOLD-SCALE~180TNEG-BOLD-SCALE~195TBOLD-EXP~210TNEG-BOLD-EXP~225TBOLD-POSITIVE~240TBOLD-NEGATIVE~255TRUNS~{~&~A~15T~A~30T~A~45T~A~60T~A~75T~A~90T~A~105T~A~120T~A~135T~A~150T~A~165T~A~180T~A~195T~A~210T~A~225T~A~240T~A~255T~A~}" runs))


(defun answer-given ()
    "Function necessary for (run-until-condition) in experiment file. While no answer is given, the return value is nil, else the respective pressed key."
    *response*)
    (defun check-arguments (model)
    (when
        (not (or (string-equal model "pmm")
                 (string-equal model "any")))
            (format t "ERROR: This function has to be called with the model parameter \"pmm\" or \"any\".")
            (return-from check-arguments nil))
    (setf *model* model))


(defun ignore (var)
 "This function does nothing an can avoid 'unused variable warnings'"
 (setf var var)
)


(defun print-bold-table (bold-values)
  "This function write the predicted bold bold values in files for each task, run and vp"
    (setf new-bold  nil)
    (setf length (list-length bold-values))
    (setf formatstring "~{~&")
    (setf formatcounter 0)

    ;; write header for table
    (loop for _ in bold-values
        do (
            progn
                (ignore _)
                (setq formatstring (concatenate 'string formatstring "~A~" (write-to-string (+ 30 formatcounter)) "T" ))
                (setf formatcounter (+ formatcounter 30))))
    (setf formatstring (concatenate 'string formatstring "~}"))

    ;; write values
    (loop with y-counter = 0 for y in (nth 0 bold-values)
        do (
            progn
                (setf x-list Nil)
                (ignore y)
                (loop for x in bold-values
                    do (
                        progn
                        (if x-list
                            (setf x-list (append x-list (list (nth y-counter x))))
                            (setf x-list (list (nth y-counter x))))))
                (if x-list
                    (progn
                        (format t formatstring x-list)))
                (setf y-counter (+ y-counter 1)))))


(defun choose-vpset (n)
  "This function reads the task/training data and stores them in the *taskarray*
   It concatenates the tasks after some training tasks."
    (assert (not (= n 0)))
    (setf *taskarray* nil)
    (let ((taskno (mod n *vp-count*)))
        (when (= taskno 0) (setq taskno *vp-count*))
        ;; load the training tasks
        (load "vpsets/fourterm-training.lisp")
        (setf *taskarray* *term-presentations*)
        (if (< taskno 10)
            (setq taskno (concatenate 'string "0" (write-to-string taskno)))
            (setq taskno (write-to-string taskno))
        )
        (load (concatenate 'string "vpsets/fourstage-vpset-" taskno ".lisp"))
        (setf *taskarray* (append *taskarray* *term-presentations*))))


(defun bold-response-to-file (run vp taskid verbose start end)
  "This function writes the predicted BOLD responses for a specific trial, run and vp to a file.
   The prediction starts at 'start' and end by 'end'. To print the BOLD prediction Table it is required
   to turn on the models output at least for the print. So verbose is passed to reset the verbose
   status of the run after the table was print"
  (let (
    (run (write-to-string run))
    (vp (write-to-string vp))
    (taskid (write-to-string taskid)))

    ;; geberate output folders
    (ensure-directories-exist  (concatenate 'string "/home/rob/Desktop/actrneuro/fang09_preofficial_ACT-R_7/log/bolds/run-" run "/"))
    (ensure-directories-exist  (concatenate 'string "/home/rob/Desktop/actrneuro/fang09_preofficial_ACT-R_7/log/bolds/run-" run "/" vp "/"))
    (ensure-directories-exist  (concatenate 'string "/home/rob/Desktop/actrneuro/fang09_preofficial_ACT-R_7/log/bolds/run-" run "/" vp "/" taskid "/"))

    ;; enable model output
    (sgp-fct `(:v , t))

    ;; print table
    (with-open-file (*standard-output*
        (concatenate 'string  "/home/rob/Desktop/actrneuro/fang09_preofficial_ACT-R_7/log/bolds/run-" run "/" vp "/" taskid "/bold-response.dat")
            :direction :output
            :if-exists :supersede
            :if-does-not-exist :create)
            (print-bold-table (predict-bold-response start (+ end 5) nil)))

    ;; reset verbose
    (sgp-fct `(:v , verbose))))






(defun print-debug ()
		(progn
			(print "----------------start output debug function---------------------------")
				(print "------>Goal-status")
			(buffer-status Goal)
			(print "------>Goal-slots")
			(buffer-chunk Goal)

			(print "------>Retrieval-status")
			(buffer-status retrieval)
			(print "------>Retrieval-slots")
			(buffer-chunk retrieval)

			(print "------>Imaginal-status")
			(buffer-status imaginal)
			(print "------>Imaginal-slots")
			(buffer-chunk imaginal)

			(print "------>Visual-status")
			(buffer-status visual)
			(print "------>Visual-slots")
			(buffer-chunk visual)

			(print "------>Visual-location-status")
			(buffer-status visual-location)
			(print "------>Visual-location-slots")
			(buffer-chunk visual-location)


			(print "------>Aural-status")
			(buffer-status Aural)
			(print "------>Aural-slots")
			(buffer-chunk Aural)

			(print "------>Aural-location-status")
			(buffer-status Aural-location)
			(print "------>Aural-location-slots")
			(buffer-chunk Aural-location)
			(print "----------------end output debug function---------------------------")
			(terpri)

		)
	)

