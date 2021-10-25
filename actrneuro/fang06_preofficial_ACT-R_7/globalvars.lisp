;;; File : globalvars.lisp

;; Flags
(defparameter *intermediate-retrieval* 'no) ;; in the process conclusion phase, should the unified model be retrieved instantly after the retrieval of a term chunk?
(defvar *strategy* nil) ; variable for saving the current strategy, i.e. 'pmm or 'any. The strategy is set by starting the experiment.

(defparameter *model* nil)
(defvar *model-loaded* nil)

(defparameter *term-presentations* nil) ; will be replaced in the vpsets files
(defvar *window-visible* nil)
(defvar *modelfocus*)
(defvar *conclusionfocus*)
(defvar *premisefocus*)
(defvar *premises*) ;; stores premises according to *numberofpremises*
(defvar *runs* nil)
(defvar *model-results* nil)
(defvar *response* nil)
(defvar *firstresponse* nil)
(defvar *taskarray*)
(defvar *total-time* 0)
(defvar *first-total-time*)
(defvar *start-time*)
(defvar *trial-start-time*)
(defvar *experiment*)

(defvar *trial* 0)

(defstruct focus
  (position 1))

(setf *modelfocus* (make-focus)) ;; :position 1))
(setf *conclusionfocus* (make-focus)) ;; :position 1))
(setf *premisefocus* (make-focus))


