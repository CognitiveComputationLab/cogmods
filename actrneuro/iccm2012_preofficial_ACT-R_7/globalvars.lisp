;;; File : globalvars.lisp

(defparameter *modelversion* "latest9")

;; Flags
(defparameter *intermediate-retrieval* 'yes) ;; in the process conclusion phase, should the unified model be retrieved instantly after the retrieval of a term chunk?
(defvar *strategy* nil) ; variable for saving the current strategy, i.e. 'pmm or 'any. The strategy is set by starting the experiment.

(defparameter *model* nil)
(defvar *model-loaded* nil)

(defparameter *term-presentations* nil) ; will be replaced in the vpsets files
(defvar *window-visible* nil)
(defvar *modelfocus*)
(defvar *conclusionfocus*)
(defvar *premisefocus*)
(defvar *premises*) ;; stores premises according to *numberofpremises*
(defvar *correct1*)
(defvar *correct2*)
(defvar *runs* nil)
(defvar *model-results* nil)
(defvar *response* nil)
(defvar *firstresponse* nil)
(defvar *taskarray*)
(defvar *total-time* 0)
(defvar *first-total-time*)
(defvar *start-time*)
(defvar *start-time-model*)
(defvar *currentswitches*)
(defvar *trial-start-time*)
(defvar *experiment*)
(defvar *previous-response* nil)

(defvar *trial* 0)

(defstruct focus
  (position 1))

(setf *modelfocus* (make-focus)) ;; :position 1))
(setf *conclusionfocus* (make-focus)) ;; :position 1))
(setf *premisefocus* (make-focus))

(defstruct switches
    numberofpremises                    ; used for checking if process premises phase has to be terminated
	give-two-responses
	give-conclusion-response            ; first / second / no
	give-model-response                 ; first / second / no
    permanent-conclusions               ; do the conclusion / model disappear shortly?
    paced                               ; externally or self paced?
    display-press-space                 ; is a letter shown to indicate that a key can be pressed from now on?
    display-press-space-at-start        ; key press at the beginning to start a trial?
    key-press-at-start-without-S        ; no S is shown at start, but the space key has to be pressed
    key-press-after-process-conclusion  ; key press after processing of the conclusion?
    intermediate-key-press              ; key presses inside a premise (after the first term)?
    correct-term                        ; which key has to be pressed for Yes?
    not-correct-term                    ; which key has to be pressed for No?
    complete-premises)                  ; are the premises shown completely as A L B?

    
; ICCM 2012   
(defparameter *iccm-currentswitches* (make-switches
:numberofpremises                   3        ; the final conclusion does not count towards this number
:give-two-responses	                'yes
:give-conclusion-response			'first
:give-model-response				'second
:permanent-conclusions              'second
:paced                              'self
:display-press-space                'yes
:display-press-space-at-start       'yes
:key-press-at-start-without-S       'no
:key-press-after-process-conclusion 'no
:intermediate-key-press             'no
:correct-term                       "C"
:not-correct-term                   "N"
:complete-premises                  'no))

