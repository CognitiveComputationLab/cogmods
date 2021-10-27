;;; File : utilities.lisp

(defvar *bold-response-start*)
(defvar *bold-response-end*)



(defun move-focus-right (focus)
  (let* ((pos (+ (focus-position focus) 1)))
    (setf (focus-position focus) pos)
    (append  (list (read-from-string
            (concatenate 'string "pos"
                 (write-to-string (focus-position focus))))))))


(defun move-focus-left (focus)
  (let* ((pos (- (focus-position focus) 1)))
     (setf (focus-position focus) pos)
    (append  (list (read-from-string
            (concatenate 'string "pos"
                 (write-to-string (focus-position focus))))))))

(defun set-focus (pos focus)
  (let* ((posname (symbol-name pos))
     (num (parse-integer
           (subseq posname 3 (length posname)))))
    (setf (focus-position focus) num)
))

(defun extend-left (leftborder)
  "This function returns a new slot name for mental model chunk. Formally the
  extend-chunk-type-slots function has been used to add new slots to an existing
  chunk on demand. In the latest ACT-R version this happens automatically. "
  (let* ((posname (symbol-name leftborder))
     (newslot (concatenate 'string "pos"
                   (write-to-string (- (parse-integer (subseq posname 3 (length posname))) 1)))))
    (read-from-string newslot)))

(defun extend-right (rightborder)
  "This function returns a new slot name for mental model chunk. Formally the
  extend-chunk-type-slots function has been used to add new slots to an existing
  chunk on demand. In the latest ACT-R version this happens automatically. "
  (let* ((posname (symbol-name rightborder))
    (newslot (concatenate 'string "pos"
                (write-to-string (+ (parse-integer (subseq posname 3 (length posname))) 1)))))
    (read-from-string newslot)))

(defun print-table (runs)
        (when (string-equal *experiment* "iccm")
            (format t "~&RUN~15TVPSET~30TTRIAL~45TTASKID~60TRT1~75TCORRECT1~90TRESPONSE1~105TEXPECTED1~120TRT2~135TCORRECT2~150TRESPONSE2~165TEXPECTED2~180TDETERMINACY~195TANS~210TRT~225TLF~{~&~A~15T~A~30T~A~45T~A~60T~A~75T~A~90T~A~105T~A~120T~A~135T~A~150T~A~165T~A~180T~A~195T~A~210T~A~225T~A~}" runs))
)


(defun print-bold-table (bold-values)  
"This function write the predicted bold bold values in files for each task, run and vp"
    (setf new-bold  nil)
    (setf length (list-length bold-values))
    (setf formatstring "~{~&")
    (setf formatcounter 0)

    (loop for module-list in bold-values
        do (
            progn
                (setq formatstring (concatenate 'string formatstring "~A~" (write-to-string (+ 30 formatcounter)) "T" ))
                (setf formatcounter (+ formatcounter 30))))
    (setf formatstring (concatenate 'string formatstring "~}"))

    (setf y-counter 0)
    (loop for y in (nth 0 bold-values)
        do (
            progn
                (setf x-list Nil)
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
    (assert (not (= n 0)))
    (setf *taskarray* nil)
    (let ((taskno (mod n 24)))
        (when (= taskno 0) (setq taskno 24))
        (load "vpsets/fourterm-training.lisp")
        (setf *taskarray* *term-presentations*)
        (if (< taskno 10)
            (setq taskno (concatenate 'string "0" (write-to-string taskno)))
            (setq taskno (write-to-string taskno))
        )
        (load (concatenate 'string "vpsets/fourstage-vpset-" taskno ".lisp"))
        (setf *taskarray* (append *taskarray* *term-presentations*))
    )
)

(defun answer-given (&n)
    (if (> (- (get-time) *trial-start-time*) 40000)
        "NA"
    *response*)) 

(defun check-arguments (model)
    (when
        (not (or (string-equal model "pmm")
                 (string-equal model "any")))
            (format t "ERROR: This function has to be called with the model parameter \"pmm\" or \"any\".")
            (return-from check-arguments nil))
    (setf *model* model))


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

    (do ((vpset 1 (+ vpset 1)))
        ((> vpset 24))

        (setf *runs* nil)

        


        (dotimes (count n)
        (let* ((run (+ 1 count))
                ;; run a complete experiment
                (result (do-experiment model :run run :vpset vpset :ans ans :rt rt :lf lf
                                        :bold-scale bold-scale :neg-bold-scale neg-bold-scale :bold-exp bold-exp
                                        :neg-bold-exp neg-bold-exp :bold-positive bold-positive :bold-negative bold-negative
                                            :verbose verbose :run-amount n)))

        (setf *runs* (append *runs* result))
        (format t "~&------------------------------> RUN ~d: VPSET ~d: ANS: ~d RT: ~d LF: ~d <------------------------------~%" run vpset ans rt lf)))

        (let ((filename (concatenate 'string  "./log/" model "/iccm_" (write-to-string (get-universal-time)) "_" (write-to-string vpset) "_" *modelversion* ".dat")))
      (with-open-file (*standard-output* filename
                         :direction :output
                         :if-exists :supersede
                         :if-does-not-exist :create)
        (print-table *runs*))

      (format t "~%File ~s written.~%" filename)))
    (format t "~%")
)



(defun print-debug ()

		(progn

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


			(print "----------------end output debug function---------------------------")
			(terpri)

		)

)





