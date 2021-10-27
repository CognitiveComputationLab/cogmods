;; (load "/home/rob/Desktop/ACTR/actr6/load-act-r-6.lisp")

(defun parse-to-float(line)
  (with-input-from-string (s line)
    (loop
      :for num := (read s nil nil)
      :while num
      :collect num)))





(load "/home/rob/Desktop/ACTR/actr7/load-act-r.lisp")
(load "/home/rob/Desktop/actrneuro/fang06_preofficial_ACT-R_7/load-fang06.lisp")


(setf runs (nth 0 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf model (nth 1 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf ans (nth 2 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf rt (nth 3 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf lf (nth 4 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf bold-scale (nth 5 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf neg-bold-scale	 (nth 6 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf bold-exp (nth 7 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf neg-bold-exp (nth 8 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf bold-positive (nth 9 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf bold-negative (nth 10 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*))
(setf verbose (search "t" (nth 11 *UNPROCESSED-COMMAND-LINE-ARGUMENTS*)))


(do-n       (parse-integer runs)
            model
           (nth 0 (parse-to-float ans))
           (nth 0 (parse-to-float rt))
           (nth 0 (parse-to-float lf))
           (nth 0 (parse-to-float bold-scale))
           (nth 0 (parse-to-float neg-bold-scale))
           (nth 0 (parse-to-float bold-exp))
           (nth 0 (parse-to-float neg-bold-exp))
           (nth 0 (parse-to-float bold-positive))
           (nth 0 (parse-to-float bold-negative))
           verbose)


(print "closing lisp programm")

(quit)



