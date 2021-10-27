;;; File : latest9.lisp
(clear-all)

(define-model nterm

 (sgp
      :v nil
      :esc t
      :ncnar nil
      :bll .5
      :crt nil
      :unstuff-visual-location nil
      :ol nil
      :bold-inc 1
      :show-focus t :needs-mouse nil :trace-detail high ;medium ;high
      :act nil
      :crt nil
      )

 (chunk-type-fct `(experiment
 trial
 state
 ))

 (chunk-type mentalmodel
 pos1
 (modeltype unified) ; {unified, conclusion, premise, annotated-premise}
 (leftborder pos1)     ; the position of the leftmost position with a non-nil value
 (rightborder pos1)    ; the position of the rightmost position with a non-nil value
 leftborderterm        ; this slot contains the term at the leftmost position with a non-nil value
 rightborderterm       ; this slot contains the term at the rigchtmost position with a non-nil value
 (modelsize 1)            ; remembers the model's size
 tbm                   ; to be merged? for discontinuous premise presentations

 ;; The following slot is only used for model type 'premise'
 initial-term          ; previously stored in the relation slot
 carry                 ; this slot is used for carrying a term that does not have a reference point yet. Example: Premise 1: A l B, C r B. Here C would be placed into the carry slot

 ;; The following slots are only used for annotated-premise
 refo                  ;; ref-obj ; refers to the term that has been seen in the previous premise
 loco                  ;; rel-obj ; refers to the ambiguous term, the term that is referenced by the annotation
 reference-term-pos    ; position of the reference object in the annotated premise
 type                  ; type of annotation: initial or inherited
 position              ; Remember position of the annotated term in the unified model; saves time in case the mental model has not been changed in the
                       ; variation phase terms can immediately be interchanged. If the model has been merged or the wrong one has
                       ; been retireved, position and loco do not match, only then the loco in the unified model has to be searched.
 trial                 ; trial number
; slots for presented terms, when other terms are presented, add slots here
 k
 x
 n
 z
 model
 )
 ;; Annotated premise chunks have the slots LOCO (to be located object), REFO (reference object)
 ;; example: Unified model: AB, 2nd premise: A is left of C -> LOCO: C, REFO: A

 (chunk-type-fct `(reasoning-task
  (strategy , *strategy*) ;; pmm or any
  phase            ; for the current phase
  step             ; status variable
  trial            ; trial number
  variation        ; will be set to yes, when annotated premises are used for the variation process
  (annotations no) ; do we have annotations?
  number           ; counts the premises. value is current premise
  direction        ; direction for moving the focus: left, right, stop
  leftborder       ; remember the leftborder, ...
  rightborder      ; remember the rightborder, ...
  leftborderterm   ; remember the leftborderterm, ...
  rightborderterm  ; remember the rightborderterm, ...
  modelsize        ; ...and the size for more precise retrievals
  conclusionsize   ; more precise retrieval of the conclusion
  focus            ; position that is focussed in the mentalmodel
  second-focus     ; position that is focussed in the conclusion
  searchterm       ; remember the term that is searched in the unified model.
  (continuous yes) ; is the task a discontinuous one? if yes, the slot value is 'no
  answer-allowed   ; model has already seen the A for "answer now" and remembers that it has not to search for this signal again
  retry-model
   time
  pmmleftborderterm    ; used for model validation
  pmmrightborderterm   ; used for model validation
  (intermediateretrieval yes)
 ))




(chunk-type term value name)

 (add-dm
    (t1 isa experiment trial 0 state attend)
    (attend isa chunk)
    (encode isa chunk)
    (start isa chunk)
    (pos-2 isa chunk)
    (pos-1 isa chunk)
    (pos0 isa chunk)
    (pos1 isa chunk)
    (pos2 isa chunk)
    (pos3 isa chunk)
    (pos4 isa chunk)
    (seen isa chunk)
    (stop isa chunk)
    (yes isa chunk)
    (no isa chunk)
    (model isa chunk)
    (first isa chunk)
    (second isa chunk)
    (both isa chunk)
    (am isa chunk)
    (pmm isa chunk)
    (checkannotation isa chunk)
    (initial isa chunk)
    (inherited isa chunk)
    (pre-search-term isa chunk)
    (search-term isa chunk)
    (search-success isa chunk)
    (search-failure isa chunk)
    (first-term-seen isa chunk)
    (wait-for-second-term isa chunk)
    (second-term-seen isa chunk)
    (insert-unknown-term isa chunk)
    (retrieve-term isa chunk)
    (retrieve-model isa chunk)
    (compare-model-with-conclusion isa chunk)
    (consecutive-conclusion-processing isa chunk)
    (search-for-more-conclusion-terms isa chunk)
    (model-conclusion-processing isa chunk)
    (check-for-initial-annotation isa chunk)
    (search-annotated-term isa chunk)
    (check-variation isa chunk)
    (recall-unified-model isa chunk)
    (recall-other-unified-model isa chunk)
    (checkfortbm isa chunk)
    (merge-models isa chunk)
    (insert-terms isa chunk)
    (wait-for-first-term isa chunk)
    (prepare-wait-for-first-term isa chunk)
    (set-moved-focus isa chunk)
    (set-second-focus isa chunk)
    (set-second-moved-focus isa chunk)
    (prepare-space-press isa chunk)
    (process-p1 isa chunk)
    (process-p2 isa chunk)
    (process-c isa chunk)
    (process-c-and-compare isa chunk)
    (compare-terms isa chunk)
    (encode-term isa chunk)
    (store-terms isa chunk)
    (inspection isa chunk)
    (unified isa chunk)
    (conclusion isa chunk)
    (premise isa chunk)
    (annotated-premise isa chunk)
    (variation isa chunk)
    (externally isa chunk)
    (first-term-processed isa chunk)
    (reset-foci isa chunk)
    (respond isa chunk)
    (respond-with-r isa chunk)
    (respond-with-f isa chunk)
    (wait isa chunk)

    ; Value string for objects when it comes from the visual for the retrieval,
    ; Name for objects when it is used for a slot name.
    (a-term isa term value "a" name a)
    (b-term isa term value "b" name b)
    (c-term isa term value "c" name c)
    (s-term isa term value "S" name s)

 )

(set-all-base-levels 1000 0)   ; set all base levels to 1000 with creation-time 0

(goal-focus t1)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; PRODUCTIONS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; There are six cluster of different production types
;;; (1) P1: Process first premise
;;; (2) P2: Process consecutive premises
;;; (3) PC: Process conclusion
;;; (4) I: Inspect conclusion if it is consistent to the unified mental model
;;; (5) V: Variation of unified mental model if conclusion does not match and there are annotated premises
;;; (6) R: Respond with r or f



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; START EXPERIMENT WITH PRESENTING "S" ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Productions for processing a term that specifies which task should be performed by the model (reasoning or maintenance, here: reasoning)

;;; s-encode-location
;;; s-start-reasoning
;;; s-task-term-retrieval-failed-retry

;; The task term is placed into the visual-location buffer by buffer stuffing. Process the seen term.
(p s-encode-location
 =goal>
    ; isa experiment
    state attend
 =visual-location>
    isa visual-location
 ?visual>
    state free
 ==>
=goal>
 +visual>
    isa move-attention
    screen-pos =visual-location
)

;; When a textual item representing the task term has been presented/seen,  the respective chunk of type 'term is
;; retrieved. The textual item is a placeholder for any kind  of object that may be associated with that item,
;; not just terms. Here it represents the task associated with the term, namely reasoning (i.e. "S" for German "Schliessen").
(p s-start-reasoning
 =goal>
    ; isa experiment
    trial =trial
    state attend
 =visual>
    isa visual-object
    value =v
 ==>
=goal>
    trial =trial
    state start
 +retrieval>
    isa term
    value =v
 =visual>
)

;; When the "S" term could not be retrieved, try to retrieve it again.
(p s-task-term-retrieval-failed-retry
 =goal>
    isa experiment
    trial =trial
    state start
 =visual-location>
    isa visual-location
 =visual>
    isa visual-object
    value =v
 ?retrieval>
    state error
 ==>
=goal>
 +retrieval>
    isa term
    value =v
 =visual-location>
)




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; P1: PROCESS FIRST PREMISE ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Productions for reading the first premise and storing it into a mentalmodel chunk of type 'unified. The first term is
;;; directly integrated into the new mental model of type unified without the intermediate step of creating a mental model of
;;; type premise. The motivation is that at this moment there is only one possible position and there is no need to later change
;;; the type to annotated-premise as might be necessary for the subsequent premises (see P2 productions). The first term is simply
;;; inserted at position 1 and the second term is inserted relative to the first term by extending the mental model to the
;;; respective direction.  This means, if the second term is left of the first one, it will be placed on position 0, if it's
 ;; right of the first one, it will be placed on position 2.

;; When the "please press space" term is presented, move the right hand to the left arrow key.
(p p1-encode-first-term
 =goal>
    ; isa experiment
    trial =trial
    state start
 =visual-location>
    isa visual-location
 ?visual>
    - state busy
 =retrieval>
    ; isa term
    value "S"
 ==>
 !bind! =newtrial (+ 1 =trial)
+goal>
    isa reasoning-task
    phase process-p1
    number 1                ;; counts premises
    trial =newtrial
 +visual>
    isa move-attention
    screen-pos =visual-location
 =visual-location>
)


 ;; The second term of the first premise has been seen.
(p p1-encode-second-term
 =goal>
    ;isa reasoning-task
    phase process-p1
 =visual-location>
    isa visual-location
 ?visual>
    - state busy
 ?retrieval>
    buffer empty
    state free
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborderterm =lbt
    rightborderterm =lbt
 ==>
=goal>
 +visual>
    isa move-attention
    screen-pos =visual-location
 =visual-location>
 =imaginal>
)

;; When a textual item representing a term has been presented/seen, the respective chunk of type 'term is retrieved. The textual item is a placeholder for any kind
;; of object that may be associated with that item, not just terms.
(p p1-retrieve-term
 =goal>
    ; isa reasoning-task
    phase process-p1
 =visual>
    isa visual-object
    value =value
 ?retrieval>
    buffer empty
    state free
 ==>
=goal>
 +retrieval>
    ; isa term
    value =value
    - name nil
 =visual>
)

;; If the retrieval for a premise term was not successful, retry.
(p p1-retrieve-term-retry
 =goal>
    ; isa reasoning-task
    phase process-p1
 =visual>
    isa visual-object
    value =value
 ?retrieval>
    state error
 ==>
=goal>
 +retrieval>
    ; isa term
    value =value
   - name nil
 =visual>
)


; Like "p1-create-mentalmodel-and-insert-first-term-intermediate-key-press" but without an intermediate key press between the terms
(p p1-create-mentalmodel-and-insert-first-term
 =goal>
    ; isa reasoning-task
    phase process-p1
 ?imaginal>
    buffer empty
 =retrieval>
    ; isa term
    - name nil
    name =name
 =visual-location>
    isa visual-location
 =visual>
    isa visual-object
 ==>
   !eval! (set-focus (read-from-string "pos1") *modelfocus*)
 =goal>
    focus pos1
    step nil
 +imaginal>
    isa mentalmodel
    pos1 =name
    leftborder pos1
    rightborder pos1
    leftborderterm =name
    rightborderterm =name
    =name yes
    !output! (=name po1 po1 =name =name)
)



 ;; There is already a mentalmodel chunk in the imaginal buffer; just add the second term. The second term has been seen on the left (< 150)
;; hence insert left of first term
(p p1-insert-second-term-left-of-first-term
 =goal>
    ; isa reasoning-task
    phase process-p1
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =lb
    modelsize =size
 =retrieval>
    ; isa term
    - name nil
    name =name
 =visual-location>
    isa visual-location
    < screen-x 150
 =visual>
    isa visual-object
    value =value
 ==>
!bind! =newlb (extend-left =lb)
 !bind! =newfocus (first (move-focus-left *modelfocus*))
 !bind! =newsize (+ =size 1)
 =goal>
    direction right
    focus =newfocus
 =imaginal>
    =newfocus =name
    leftborder =newlb
    leftborderterm =name
    modelsize =newsize
    =name yes
)

 ;; There is already a mentalmodel chunk in the imaginal buffer; just add the second term. The second term has been seen on the right (> 150)
;; hence insert right  of first term
(p p1-insert-second-term-right-of-first-term
 =goal>
    ; isa reasoning-task
    phase process-p1
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =lb
    modelsize =size
 =retrieval>
    ; isa term
    - name nil
    name =name
 =visual-location>
    isa visual-location
    > screen-x 150
 =visual>
    isa visual-object
    value =value
 ==>
   !bind! =newrb (extend-right =lb)
    !bind! =newfocus (first (move-focus-right *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    focus =newfocus
    direction left
 =imaginal>
    =newfocus =name
    rightborder =newrb
    rightborderterm =name
    modelsize =newsize
    =name yes
)




;; Self-paced: The first premise has been integrated into the mentalmodel chunk; start processing the second premise. For all other consecutive
;; premises there is no mental model in the imaginal buffer but in the retrieval buffer and has to be transferred to the imaginal
;; first . There are diferent kinds of consecutive premises: Those that trigger (1) continuous, (2) semi-continuous, and (3) discontinuous
;; premises (see below).
(p p1-process-first-premise-complete
 =goal>
    ; isa reasoning-task
    phase process-p1
    number =num
    direction =direction
    trial =trial
    focus =focus
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    - rightborder =lb
    rightborder =rb
    leftborderterm =lbt
    - rightborderterm =lbt
 ==>
!bind! =newnumber (+ =num 1)
 =goal>
    phase process-p2
    number =newnumber
    direction =direction
    trial =trial
    focus =focus
    step wait-for-first-term
 +goal> =goal
 =imaginal>
 +visual>
    isa clear
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; P2: PROCESS CONSECUTIVE PREMISES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Productions for reading the first term of the second premise and store it into a mentalmodel chunk of type 'premise.


 ;; The first term of the nth premise has been seen. Continue with step nil. This production fires when the number of the already processed premises
 ;; is smaller than or equal to the number of premises in this experiment.
(p p2-encode-first-term
 =goal>
    ; isa reasoning-task
    phase process-p2
    step wait-for-first-term
 =visual-location>
    isa visual-location
 ?visual>
    state free
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ==>
=goal>
    step first-term-seen
 +visual>
    isa move-attention
    screen-pos =visual-location
 =visual-location>
 =imaginal>
)

;; see above: p1-retrieve-term. At this point, different from the first premise,  it is necessary to differentiate between the first and second term.
(p p2-release-unified-model-and-retrieve-first-term
 =goal>
    ; isa reasoning-task
    step first-term-seen
 ?retrieval>
    buffer empty
    state free
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
 =visual>
    isa visual-object
    value =value
 ==>
=goal>
    leftborder =lb
    rightborder =rb
 +retrieval>
    ; isa term
    - name nil
    value =value
    - name nil
 =visual>
 =imaginal>
)

;; If the retrieval for a premise term was not successful, retry.
(p p2-retrieve-first-term-retry
 =goal>
    ; isa reasoning-task
    step first-term-seen
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ?retrieval>
    state error
 =visual>
    isa visual-object
    value =value
 ==>
=goal>
 +retrieval>
    ; isa term
    - name nil
    value =value
    - name nil

 =visual>
 =imaginal>
)

;;; If the first term is presented - no matter if it is a known or unknown term - a new premise chunk is created in order
;;; to buffer the respective term; no integration into the current model takes place until the next term of the premise has been
;;; seen because at this moment individuals either cannot know where to integrate the unknown term, or if the type has to be changed
;;; to annotated-premise.
;;;
;;; a) Unknown term seen first
;;; A l B     A is to the left of B
;;; C r B     C is to the right of B  <- unknown letter is seen first, integration into the model not possible.
;;; C l D     C is to the left of D
;;; A l C     Is A to the left of C? (Conclusion)
;;;
;;; b) Possible to integrate second term at different positions and therefore change type to annotated-premise.
;;; A l B
;;; A l C <- Type has to be changed from premise to annotated-premise because term C can be integrated at different positions.


;;; Create premise chunk and release the unified model into the declarative memory

 ; The seen letter is the left one, so create a premise mentalmodel, place the term into the model and remember the left relation.
 ; Finally recall the unified mentalmodel for the next step: search this term in the unified model.
(p p2-first-term-found-create-premise-model-insert-left-and-recall-unified-model
 =goal>
    ; isa reasoning-task
    leftborder =lb
    rightborder =rb
    step first-term-seen

 =visual-location>
    isa visual-location
    < screen-x 150
 =retrieval>
    ; isa term
    - name nil
    name =name
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 =visual>
    isa visual-object
 ==>
=goal>
    step pre-search-term
    searchterm =name
 +imaginal>
    isa mentalmodel
    modeltype premise
    pos1 =name
    initial-term left           ; left term seen first
    leftborderterm =name
    rightborderterm =name
    =name yes
    !output! (=name)
+retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
)

 ; The seen letter is the right one, so create a premise mentalmodel, place the term into the model and remember the right relation.
 ; Finally recall the unified mentalmodel for the next step: search this term in the unified model.
(p p2-first-term-found-create-premise-model-insert-right-and-recall-unified-model
 =goal>
    ; isa reasoning-task
    leftborder =lb
    rightborder =rb
    step first-term-seen
 =visual-location>
    isa visual-location
    > screen-x 150
 =retrieval>
    ; isa term
    - name nil
    name =name
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 =visual>
    isa visual-object
 ==>
=goal>
    step pre-search-term
    searchterm =name
 +imaginal>
    isa mentalmodel
    modeltype premise
    pos1 =name
    initial-term right          ; right term seen first
    leftborderterm =name
    rightborderterm =name
    =name yes
    !output! (=name)

 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
)



 ; Finally recall the unified mentalmodel for the next step: search this term in the unified model.
(p p2-first-term-found-create-premise-model-recall-unified-model-retry
 =goal>
    ; isa reasoning-task
    leftborder =lb
    rightborder =rb
    step pre-search-term
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 ==>
=goal>
 =imaginal>
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
)


;;; Check if the seen term is at the already focussed position


;; Focus is already at the position of the newly seen term. Prepare for processing the second term of this premise.
(p p2-presearch-check-if-focus-is-already-at-correct-position-first-term-left-first-term-match
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term left
    carry nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =f =v
 ==>
=goal>
    step wait-for-second-term
    direction right
 =imaginal>
 =retrieval>
)


;; Focus is already at the position of the newly seen term. Prepare for processing the second term of this premise.
(p p2-presearch-check-if-focus-is-already-at-correct-position-first-term-right-first-term-match
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term right
    carry nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =f =v
 ==>
=goal>
    step wait-for-second-term
    direction left
 =imaginal>
 =retrieval>
)

;; Focus is at position of the second term. Prepare for inserting the first (unknown) term of this premise.
(p p2-presearch-check-if-focus-is-already-at-correct-position-first-term-left-second-term-match
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term left
    carry =ca
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =f =v
 ==>
=goal>
    step search-success
    direction stop
 =imaginal>
 =retrieval>
)

;; Focus is at position of the second term. Prepare for inserting the first (unknown) term of this premise.
(p p2-presearch-check-if-focus-is-already-at-correct-position-first-term-right-second-term-match
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term right
    carry =ca
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =f =v
 ==>
=goal>
    step search-success
    direction stop
 =imaginal>
 =retrieval>
)

;; PRESEARCH 4 productions modified

;; Search for term, continue moving through the model.
(p p2-presearch-check-if-focus-is-already-at-correct-position-first-term-right-no-match-at-left-border
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =f =v
    leftborder =f
    =v yes
 ==>
!bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    step search-term
    direction right
    focus =newfocus
 =imaginal>
 =retrieval>
)

;; Search for term, continue moving through the model.
(p p2-presearch-check-if-focus-is-already-at-correct-position-first-term-right-no-match-and-not-at-left-border
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =f =v
    - leftborder =f
    =v yes
 ==>
=goal>
    step search-term
    focus nil
    direction right
 =imaginal>
 =retrieval>
)


;; Search for term, continue moving through the model.
(p p2-presearch-check-if-focus-is-already-at-correct-position-second-term-no-match-searchterm-not-in-unified-model
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
    number =num
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    pos1 =ca
    carry =carry
    leftborder =lb
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =f =v
    =v nil
 ==>
   !bind! =newnumber (+ =num 1)
    !eval! (set-focus =lb *modelfocus*)
 =goal>
    step tbm-1
    continuous no
 =imaginal>
 =retrieval>
    tbm yes
)

;; Presearch Change 2

;; Search for term, continue moving through the model.
(p p2-presearch-check-if-focus-is-already-at-correct-position-first-term-right-no-match-at-left-border-searchterm-not-in-unified-model
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    pos1 =ca
    carry nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =f =v
    leftborder =f
    =v nil
 ==>
=goal>
    step wait-for-second-term
    direction right
 =imaginal>
    carry =ca
 =retrieval>
)


;; Search for term, continue moving through the model.
(p p2-presearch-check-if-focus-is-already-at-correct-position-first-term-right-no-match-and-not-at-left-border-searchterm-not-in-unified-model
 =goal>
    ; isa reasoning-task
    step pre-search-term
    focus =f
    searchterm =v
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    pos1 =ca
    carry nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =f =v
    - leftborder =f
    =v nil
 ==>
=goal>
    step wait-for-second-term
    direction right
 =imaginal>
    carry =ca
 =retrieval>
)

;;; Productions for searching the presented term in the unified model

; If the focus is at the rightborder of the unified model, set search direction to left.
(p p2-search-term-set-focus
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    focus nil
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    focus =lb
    direction right
 =retrieval>
=imaginal>
)

; If the focus is at the rightborder of the unified model, set search direction to left.
(p p2-search-term-set-direction-left
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    focus =focus
    direction stop
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    rightborder =focus
 ==>
=goal>
    direction left
 =retrieval>
 =imaginal>
)

; If the focus is at the leftborder of the unified model, set search direction to right.
(p p2-search-term-set-direction-right
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    focus =focus
    direction stop
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
 ==>
=goal>
    direction right
 =retrieval>
  =imaginal>
)

;; If the focus is at the left border and the to be searched term is not at this position, set the focus to the term next to it at the right.
(p p2-search-term-at-left-border-focus-right-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    direction right
    focus =pos
    searchterm =value
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =pos =value
    leftborder =pos
    rightborder =rb
 ==>
!bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
)

;; If the focus is at the right border and the to be searched term is not at this position,  set the focus to the term next to it at the left.
(p p2-search-term-at-right-border-focus-left-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    direction left
    focus =pos
    searchterm =value
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    rightborder =pos
    - =pos =value
 ==>
!bind! =newfocus (first (move-focus-left *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
)

 ;; Focus is in between, not at left or right border.
(p p2-search-term-move-left-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    direction left
    focus =pos
    searchterm =value
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - leftborder =pos
    - =pos =value
 ==>
!bind! =newfocus (first (move-focus-left *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
)

 ;; Focus is in between, not at left or right border.
(p p2-search-term-move-right-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    direction right
    focus =pos
    searchterm =value
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - rightborder =pos
    - =pos =value
 ==>
!bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
)

;;; Productions for completing term search. If a search fo a term was successful there are two possible
;;; types: (1) The term is the firsrt term of the premise, (2) The term is the second term of the premise.
;;; In (2) in addition the current mental model is extended to the respective direction.

;; The to be searched term was found; set step to search-success.
(p p2-search-term-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    focus =pos
    searchterm =value
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =pos =value
 ==>
=goal>
    step search-success
    direction stop
 =retrieval>
)


; Like "p2-search-success-first-term-right-intermediate-key-press" without intermediate key press.
(p p2-search-success-first-term-right
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-success
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term right
    carry nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
 ==>
=goal>
    step wait-for-second-term
    direction left
 =imaginal>
 =retrieval>
)


; Like "p2-search-success-first-term-left-intermediate-key-press" without intermediate key press.
(p p2-search-success-first-term-left
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-success
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term left
    carry nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    rightborder =rb
 ==>
=goal>
    step wait-for-second-term
    direction right
 =imaginal>
 =retrieval>
)

;; If the term searched for in the unified model was the second term of the current premise, continue with inserting the first term that previously
;; has not been found in the current model.
(p p2-search-success-secondterm-left-extend-left
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-success
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term left
    carry =ca
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
 ==>
!bind! =newlb (extend-left =lb)
 !bind! =newfocus (first (move-focus-left *modelfocus*))
 =goal>
    step insert-unknown-term
    focus =newfocus
    direction left
 =imaginal>
    carry nil
 =retrieval>
    leftborder =newlb
    =newlb nil
)


;; If the term searched for in the unified model was the second term of the current premise, continue with inserting the first term that previously
;; has not been found in the current model.
(p p2-search-success-secondterm-right-extend-right
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-success
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term right
    carry =ca
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    rightborder =rb
 ==>
!bind! =newrb (extend-right =rb)
 !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    step insert-unknown-term
    focus =newfocus
    direction right
 =imaginal>
    carry nil
 =retrieval>
    rightborder =newrb
    =newrb nil
)

;;; Productions for completing term search.

;; Focus position is left border of the mental model, seen term doesn't match, focus direction is 'left.
(p p2-search-term-at-left-border-focus-left-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    direction left
    focus =lb
    searchterm =value
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =lb =value
    leftborder =lb
 ==>
=goal>
    step search-failure
    direction stop
 =retrieval>
)

;; Focus position is right border of the mental model, seen term doesn't match, focus direction is 'right.
(p p2-search-term-at-right-border-focus-right-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-term
    direction right
    focus =rb
    searchterm =value
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =rb =value
    rightborder =rb
 ==>
=goal>
    step search-failure
    direction stop
 =retrieval>
)



;; Like "p2-search-first-term-failure-intermediate-key-press"  without intermediate key press.
(p p2-search-first-term-failure
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-failure
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    pos1 =ca
    carry nil
 ==>
=goal>
    step wait-for-second-term
 =imaginal>
    carry =ca
)

;; Discontinuous: (1) Mark the old unified model as "to be merged". (2) Take the new premise as a new unified model.(3) Continue processing the premises.
(p p2-search-second-term-failure-discontinuous-premise-order-set-tbm
 =goal>
    ; isa reasoning-task
    phase process-p2
    step search-failure
    number =num
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lb
    carry =ca
 ==>
=goal>
    step tbm-1
    continuous no
 =retrieval>
    tbm yes
 =imaginal>
)

;; Discontinuous: (1) Mark the old unified model as "to be merged". (2) Take the new premise as a new unified model.(3) Continue processing the premises.
(p p2-search-second-term-failure-discontinuous-premise-order-release-tbm-model
 =goal>
    ; isa reasoning-task
    phase process-p2
    step tbm-1
    number =num
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lbm
    rightborder =rbm
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lb
    carry =ca
 ==>
=goal>
    step tbm-2
    leftborder =lbm
    rightborder =rbm
 =imaginal>
 -retrieval>
)


;; Discontinuous: (1) Mark the old unified model as "to be merged". (2) Take the new premise as a new unified model.(3) Continue processing the premises.
(p p2-search-second-term-failure-discontinuous-premise-order-rehearse-tbm-model
 =goal>
    ; isa reasoning-task
    phase process-p2
    step tbm-2
    number =num
    leftborder =lbm
    rightborder =rbm
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lb
    carry =ca
 ==>
=goal>
    step tbm-3
 =imaginal>
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lbm
    rightborder =rbm
    tbm yes
)

;; Discontinuous: (1) Mark the old unified model as "to be merged". (2) Take the new premise as a new unified model.(3) Continue processing the premises.
(p p2-search-second-term-failure-discontinuous-premise-order-rehearse-tbm-model-retry
 =goal>
    ; isa reasoning-task
    phase process-p2
    step tbm-3
    number =num
    leftborder =lbm
    rightborder =rbm
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lb
    carry =ca
 ==>
=goal>
 =imaginal>
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lbm
    rightborder =rbm
    tbm yes
)

;; Discontinuous: (1) Mark the old unified model as "to be merged". (2) Take the new premise as a new unified model.(3) Continue processing the premises.
(p p2-search-second-term-failure-discontinuous-premise-order-release-tbm-model-again
 =goal>
    ; isa reasoning-task
    phase process-p2
    step tbm-3
    number =num
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lbm
    rightborder =rbm
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lb
    carry =ca
 ==>
=goal>
    step tbm-4
 =imaginal>
 -retrieval>
)


;; Discontinuous: (1) Mark the old unified model as "to be merged". (2) Take the new premise as a new unified model.(3) Continue processing the premises.
(p p2-search-second-term-failure-discontinuous-premise-order-send-copy-of-premise-to-declarative
 =goal>
    ; isa reasoning-task
    phase process-p2
    step tbm-4
    number =num
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lb
    carry =ca
 ==>
=goal>
    step tbm-5
    !output! =imaginal
 +imaginal> =imaginal
)

;; Discontinuous: (1) Mark the old unified model as "to be merged". (2) Take the new premise as a new unified model.(3) Continue processing the premises.
(p p2-search-second-term-failure-discontinuous-premise-order-set-type-of-premise-to-unified-model-and-set-focus-to-left-border-of-model
 =goal>
    ; isa reasoning-task
    phase process-p2
    step tbm-5
    number =num
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lb
    carry =ca
 ==>
   !bind! =newnumber (+ =num 1)
    !eval! (set-focus =lb *modelfocus*)
 =goal>
    step prepare-wait-for-first-term
    number =newnumber
    focus =lb
 =imaginal>
    modeltype unified
)



;;; Productions for processing and integrating the second term into the premise.

;; When the second term of the nth premise has been seen set step to second-term-seen.
(p p2-encode-second-term
 =goal>
    ; isa reasoning-task
    phase process-p2
    step wait-for-second-term
 =visual-location>
    isa visual-location
 ?visual>
    state free
 ==>
=goal>
    step second-term-seen
    searchterm nil
 +visual>
    isa move-attention
    screen-pos =visual-location
 =visual-location>
)

;; see above: p1-retrieve-term, p2-retrieve-first-term
(p p2-retrieve-second-term
 =goal>
    ; isa reasoning-task
    step second-term-seen
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
 =visual>
    isa visual-object
    value =value
 ==>
=goal>
    leftborder =lb
    rightborder =rb
 +retrieval>
    ; isa term
    - name nil
    value =value
 =visual>

)

;; If the retrieval for a premise term was not successful, retry.
(p p2-retrieve-second-term-retry
 =goal>
    ; isa reasoning-task
    step second-term-seen
 ?retrieval>
    state error
 =visual>
    isa visual-object
    value =value
 ==>
=goal>
 +retrieval>
    ; isa term
    - name nil
    value =value
 =visual>
)

;; If the second term is the unknown term, insert it into the premise chunk and retrieve the current mental model.
(p p2-second-term-insert-right-into-premise
 =goal>
    ; isa reasoning-task
    phase process-p2
    step second-term-seen
    leftborder =lb
    rightborder =rb
 =visual-location>
    isa visual-location
    > screen-x 150
 =retrieval>
    ; isa term
    - name nil
    name =name
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term left           ; left term was seen first
    rightborder =rbp
    leftborder =rbp
    modelsize =size
 =visual>
    isa visual-object
 ==>
!bind! =dummy (set-focus =rbp *premisefocus*)
 !bind! =newrbp (extend-right =rbp)
 !bind! =newsize (+ =size 1)
 =goal>
    searchterm =name
    step recall-unified-model
 =imaginal>
    =newrbp =name
    rightborder =newrbp
    rightborderterm =name
    modelsize =newsize
    =name yes
)

;; If the second term is the unknown term, insert it into the premise chunk and retrieve the current mental model.
(p p2-second-term-insert-left-into-premise
 =goal>
    ; isa reasoning-task
    phase process-p2
    step second-term-seen
    leftborder =lb
    rightborder =rb
 =visual-location>
    isa visual-location
    < screen-x 150
 =retrieval>
    ; isa term
    - name nil
    name =name
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    initial-term right          ; right term was seen first
    leftborder =lbp
    rightborder =lbp
    modelsize =size
 =visual>
    isa visual-object
 ==>
!bind! =dummy (set-focus =lbp *premisefocus*)
 !bind! =newlbp (extend-left =lbp)
 !bind! =newsize (+ =size 1)
 =goal>
    searchterm =name
    step recall-unified-model
 =imaginal>
    =newlbp =name
    leftborder =newlbp
    leftborderterm =name
    modelsize =newsize
    =name yes
)

(p p2-second-term-recall-unified-model
 =goal>
    ; isa reasoning-task
    phase process-p2
    step recall-unified-model
    leftborder =lb
    rightborder =rb
 ?retrieval>
    buffer empty
    state free
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    carry =ca
 ==>
=goal>
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
    =ca nil
 =imaginal>
)

(p p2-second-term-recall-unified-model-2
 =goal>
    ; isa reasoning-task
    phase process-p2
    step recall-unified-model
    leftborder =lb
    rightborder =rb
 ?retrieval>
    buffer empty
    state free
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    carry nil
 ==>
=goal>
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
 =imaginal>
)

(p p2-second-term-recall-unified-model-retry
 =goal>
    ; isa reasoning-task
    phase process-p2
    step recall-unified-model
    leftborder =lb
    rightborder =rb
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    carry =ca
 ==>
=goal>
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
    =ca nil
 =imaginal>
)

(p p2-second-term-recall-unified-model-2-retry
 =goal>
    ; isa reasoning-task
    phase process-p2
    step recall-unified-model
    leftborder =lb
    rightborder =rb
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    carry nil
 ==>
=goal>
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
 =imaginal>
)

(p p2-second-term-unified-model-found
 =goal>
    ; isa reasoning-task
    phase process-p2
    step recall-unified-model
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 ==>
=goal>
    step second-term-seen
 =retrieval>
 =imaginal>
)

;; If the first term of the premise has not been found in the unified model start the search for the second term of the premise.
(p p2-first-term-not-found-start-second-term-search
 =goal>
    ; isa reasoning-task
    phase process-p2
    step second-term-seen
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lb
    - rightborder =lb
    carry =ca
 ?retrieval>
    - state busy
 ==>
=goal>
    step pre-search-term
    direction stop
 =imaginal>
)

;;; Productions initiating the insertion process for the second term of the premise into the unified model. The first term is known.

;; Extend the mental model for insertion of the new term and move the current model focus in the respective direction.
(p p2-move-focus-left-to-insertion-position-of-second-term-extend-left
 =goal>
    ; isa reasoning-task
    phase process-p2
    step second-term-seen
    direction left
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp
    - rightborder =lbp
    carry nil
 =retrieval>
    ; isa mentalmodel
    leftborder =lb
 ==>
!bind! =newlb (extend-left =lb)
 !bind! =newfocus (first (move-focus-left *modelfocus*))
 =goal>
    focus =newfocus
    step insert-unknown-term
 =imaginal>
 =retrieval>
    leftborder =newlb
    =newlb nil
)

;; Extend the mental model for insertion of the new term and move the current model focus in the respective direction.
(p p2-move-focus-right-to-insertion-position-of-second-term-extend-right
 =goal>
    ; isa reasoning-task
    phase process-p2
    step second-term-seen
    direction right
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    rightborder =rbp
    - leftborder =rbp
    carry nil
 =retrieval>
    ; isa mentalmodel
    rightborder =rb
 ==>
!bind! =newrb (extend-right =rb)
 !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
    step insert-unknown-term
 =imaginal>
 =retrieval>
    rightborder =newrb
    =newrb nil
)


;;; Productions for both the first term is known or unknown. The respective unknow term is inserted. The known term is already in the unified model.
;;; The move direction is left.


;; Indeterminate: The unknown term should be inserted left, but there is already a term at this position,
;; so a annotated premise is created for remembering that there is an uncertainty of the insertion of the new term.
(p p2-move-left-add-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction left
    trial =trial
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =pos nil
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp     ; left term has to be inserted
    rightborder =rbp
    =lbp =lbv           ; value of left term has to be inserted
    =rbp =rbv
    modelsize =size
 ==>
!bind! =newfocus (first (move-focus-left *modelfocus*))
 =goal>
    annotations yes
    focus =newfocus
 =retrieval>
 +imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    leftborder =lbp
    rightborder =rbp
    =lbp =lbv               ; lbv is left of rbv
    =rbp =rbv
    leftborderterm =lbv
    rightborderterm =rbv
    loco =lbv
    refo =rbv
    reference-term-pos right ; refo is right of loco
    type initial
    ;;trial =trial ; Restrict to current trial
    modelsize =size
    !output! (=lbp =rbp =lbv =rbv right =size)
)

;; If there is already an annotation no further annotation is added.
(p p2-move-left-add-no-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction left
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =pos nil
 =imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
 ==>
!bind! =newfocus (first (move-focus-left *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
 =imaginal>
)

;; There is an annotation in the imaginal buffer and the focus is on an empty position; the unknown term can be integrated into the unified model.
(p p2-moved-left-found-empty-pos-annotation-in-imaginal
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction left
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =pos nil
    modelsize =size
 =imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    loco =value
 ==>
!bind! =newsize (+ =size 1)
 =goal>
    direction stop
 =retrieval>
    leftborder =pos
    =pos =value
    leftborderterm =value
    modelsize =newsize
    =value yes
 =imaginal>
    position =pos
    !output! =imaginal

 +imaginal> =imaginal
)

;; There is a premise in the imaginal buffer and the focus is on an empty position; the unknown term can be integrated into the unified model.
(p p2-moved-left-found-empty-pos-premise-in-imaginal
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction left
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =pos nil
    rightborderterm =rb
    modelsize =size
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp
    =lbp =value
 ==>
!bind! =newsize (+ =size 1)
 =goal>
    direction left
    step checkannotation
    leftborder =value
    rightborder =rb
    modelsize =size
 =retrieval>
    leftborder =pos
    =pos =value
    leftborderterm =value
    modelsize =newsize
    =value yes
 =imaginal>
)

;; We know that there are no annotations because annotaion-slot of reasoning-task chunk has value 'no; we skip search for annotations.
;; Recall of model not necessary because it has not been cleared from retrieval buffer.
(p p2-moved-left-no-search-for-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkannotation
    annotations no
    direction left
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    rightborder =rbp
    =rbp =r
 ==>
=goal>
    step insert-unknown-term
    direction stop
 =retrieval>
 =imaginal>
    carry nil
)

;; We know that there are annotations because annotaion-slot of reasoning-task chunk has value 'yes.
;; There is a premise is in the imaginal buffer and we start a retrieval for an annotation.
(p p2-moved-left-search-for-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkannotation
    annotations yes
    trial =trial
    direction left
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    rightborder =rbp
    =rbp =r
 ==>
=goal>
    leftborder =lb
    rightborder =rb
 +retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    loco =r
    ;;trial =trial ; Restrict to current trial
 =imaginal>
)

;; The annotaion-search failed, the retrieval module is in error state, in the imaginal is a premise, we start the retrieval for a recently
;; released unified model.
(p p2-moved-left-no-annotation-found-recall-unified-model
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkannotation
    leftborder =lb
    rightborder =rb
    direction left
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 ==>
=goal>
    step insert-unknown-term
    direction stop
 +retrieval>
    ; isa mentalmodel
    leftborder =lb
    rightborder =rb
    modeltype unified
 =imaginal>
    carry nil
)

;; The annotaion-search succeeded, in the imaginal is a premise, we start the retrieval for a recently
;; released unified model and create a inherited annotation.
(p p2-moved-left-annotation-found-create-inherited-annotation-and-recall-unified-model
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkannotation
    trial =trial
    leftborder =lb
    rightborder =rb
    direction left
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    modelsize =size
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp
    rightborder =rbp
    =lbp =l
    =rbp =r
 ==>
=goal>
    step insert-unknown-term
    direction stop
 +retrieval>
    ; isa mentalmodel
    leftborder =lb
    rightborder =rb
    modeltype unified
 +imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    leftborder =lbp
    rightborder =rbp
    leftborderterm =l
    rightborderterm =r
    =lbp =l
    =rbp =r
    refo =r
    loco =l ;; new ambiguous term
    reference-term-pos right
    type inherited
    ;;trial =trial ; Restrict to current trial
    modelsize =size
    carry nil
    !output! (=lbp =rbp =l =r =size)
)

;;; Productions for both the first term is known or unknown. The respective unknow term is inserted. The known term is already in the unified model.
;;; The move direction is right.

;; Indeterminate: The unknown term should be inserted right, but there is already a term at this position,
;; so a annotated premise is created for remembering that there is an uncertainty of the insertion of the new term.
(p p2-move-right-add-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction right
    trial =trial
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =pos nil
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp
    rightborder =rbp ; right term has to be inserted
    =lbp =lbv
    =rbp =rbv ; value of right term has to be inserted
    modelsize =size
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    annotations yes
    focus =newfocus
 =retrieval>
 +imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    leftborder =lbp
    rightborder =rbp
    =lbp =lbv  ; lbv is left of rbv
    =rbp =rbv
    leftborderterm =lbv
    rightborderterm =rbv
    loco =rbv
    refo =lbv
    reference-term-pos left ; refo is left of loco
    type initial
    ;;trial =trial ; Restrict to current trial
    modelsize =size
)
;; If there is already an annotation no further annotation is added.
(p p2-move-right-add-no-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction right
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =pos nil
 =imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
 =imaginal>
)

;; There is an annotation in the imaginal buffer and the focus is on an empty position; the unknown term can be integrated into the unified model.
(p p2-moved-right-found-empty-pos-annotation-in-imaginal
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction right
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =pos nil
    modelsize =size
 =imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    loco =value
 ==>
   !bind! =newsize (+ =size 1)
 =goal>
    direction stop
 =retrieval>
    rightborder =pos
    =pos =value
    rightborderterm =value
    modelsize =newsize
    =value yes
 =imaginal>
    position =pos
 +imaginal> =imaginal
)

;; There is a premise in the imaginal buffer and the focus is on an empty position; the unknown term can be integrated into the unified model.
(p p2-moved-right-found-empty-pos-premise-in-imaginal
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction right
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =pos nil
    leftborderterm =lb
    modelsize =size
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    rightborder =rbp
    =rbp =value
 ==>
   !bind! =newsize (+ =size 1)
 =goal>
    direction right
    step checkannotation
    rightborder =value
    leftborder =lb
    modelsize =size
 =retrieval>
    rightborder =pos
    =pos =value
    rightborderterm =value
    modelsize =newsize
    =value yes
 =imaginal>
)

;; We know that there are no annotations because annotaion-slot of reasoning-task chunk has value 'no; we skip search for annotations.
;; Recall of model not necessary because it has not been cleared from retrieval buffer.
(p p2-moved-right-no-search-for-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkannotation
    annotations no
    direction right
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp
    =lbp =l
 ==>
=goal>
    step insert-unknown-term
    direction stop
 =retrieval>
 =imaginal>
    carry nil
)

;; We know that there are annotations because annotaion-slot of reasoning-task chunk has value 'yes.
;; There is a premise is in the imaginal buffer and we start a retrieval for an annotation.
(p p2-moved-right-search-for-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkannotation
    annotations yes
    trial =trial
    direction right
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp
    =lbp =l
 ==>
=goal>
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 +retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    loco =l
    ;;trial =trial ; Restrict to current trial
 =imaginal>
)

;; The annotaion-search failed, the retrieval module is in error state, in the imaginal is a premise, we start the retrieval for a recently
;; released unified model.
(p p2-moved-right-no-annotation-found-recall-unified-model
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkannotation
    leftborder =lb
    rightborder =rb
    direction right
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 ==>
=goal>
    step insert-unknown-term
    direction stop
 +retrieval>
    ; isa mentalmodel
    leftborder =lb
    rightborder =rb
    modeltype unified
 =imaginal>
    carry nil
)

;; The annotaion-search failed, the retrieval module is in error state, in the imaginal is a premise, we start the retrieval for a recently
;; released unified model.
(p p2-moved-right-no-annotation-found-recall-unified-model-retry
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    leftborder =lb
    rightborder =rb
    direction stop
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype premise
 ==>
=goal>
 +retrieval>
    ; isa mentalmodel
    leftborder =lb
    rightborder =rb
    modeltype unified
 =imaginal>
)

;; The annotaion-search succeeded, in the imaginal is a premise, we start the retrieval for a recently
;; released unified model and create a inherited annotation.
(p p2-moved-right-annotation-found-create-inherited-annotation-and-recall-unified-model
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkannotation
    trial =trial
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
    direction right
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    modelsize =size
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp
    rightborder =rbp
    =lbp =l
    =rbp =r
 ==>
=goal>
    step insert-unknown-term
    direction stop
 +retrieval>
    ; isa mentalmodel
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
    modeltype unified
 +imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    leftborder =lbp
    rightborder =rbp
    leftborderterm =l
    rightborderterm =r
    =lbp =l
    =rbp =r
    refo =l
    loco =r ;; new ambiguous term
    reference-term-pos left
    type inherited
    ;;trial =trial ; Restrict to current trial
    modelsize =size
)

;; The annotaion-search succeeded, in the imaginal is a premise, we start the retrieval for a recently
;; released unified model and create a inherited annotation.
(p p2-moved-right-annotation-found-create-inherited-annotation-and-recall-unified-model-retry
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    trial =trial
    leftborder =lb
    rightborder =rb
    direction stop
 ?retrieval>
    buffer empty
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    carry nil
 ==>
=goal>
    direction stop
 +retrieval>
    ; isa mentalmodel
    leftborder =lb
    rightborder =rb
    modeltype unified
 =imaginal>
)

;;; Productions for completing the processing phase of the current premise.

;; If there is a premise chunk in the imaginal buffer, transfer the unified model to the imaginal buffer.
(p p2-premise-done-release-premise-transfer-model-to-imaginal
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction stop
    number =num
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 =imaginal>
    ; isa mentalmodel
    modeltype premise
    leftborder =lbp
    - rightborder =lbp
    rightborder =rbp
    - =lbp nil
    - =rbp nil
    carry nil
 ==>
   !bind! =newnumber (+ =num 1)
 =goal>
    step checkfortbm
    number =newnumber
    second-focus nil
 +imaginal> =retrieval
)

;; If there is an annotation chunk in the imaginal buffer with value nil in the position slot, set the value to the respective position
(p p2-premise-done-set-position-in-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction stop
    number =num
    focus =pos
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 =imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    position nil
    carry nil
 ==>
   !bind! =newnumber (+ =num 1)
 =goal>
 =imaginal>
    position =pos
 =retrieval>
)

;; If there is an annotation chunk in the imaginal buffer with a non-nil value in the position slot, transfer the unified model to the imaginal buffer.
(p p2-premise-done-release-annotation-transfer-model-to-imaginal
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-unknown-term
    direction stop
    number =num
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 =imaginal>
    ; isa mentalmodel
    modeltype annotated-premise
    - position nil
    carry nil
 ==>
   !bind! =newnumber (+ =num 1)
 =goal>
    step checkfortbm
    number =newnumber
    second-focus nil
 +imaginal> =retrieval
)

;;; Productions for checking if there is a model that has to be merged (tbm) and finding a term present in both models.
;;; There a two unified models: one is in the imaginal buffer, the tbmmodel is in declarative memory and has to be tretrieved.

;; (Semi-)Continuous: There is no need to search for models that are marked with to be merged.
;; Continue with next premise.
(p p2-no-search-for-tbmmodel
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkfortbm
    continuous yes
 ==>
=goal>
    step prepare-wait-for-first-term
)

;; Discontinuous: Check if there are unified models in the declarative memory that are marked with tbm (to be merged) and have to be merged with
;; the current unified model.
(p p2-search-for-tbmmodel
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkfortbm
    continuous no
 ?retrieval>
    buffer empty
    state free
 ==>
=goal>
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    ;:recently-retrieved nil
)

;; Discontinuous: No unified model found that has to be merged with the unified one.
(p p2-search-for-tbmmodel-failure
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkfortbm
 ?retrieval>
    state error
 ==>
=goal>
    step prepare-wait-for-first-term
)

;; Discontinuous: A unified model that has to be merged was found. Now start the merging process with setting the foci to the leftborders of the two models.
(p p2-search-for-tbmmodel-success-set-first-focus
 =goal>
    ; isa reasoning-task
    phase process-p2
    step checkfortbm
    second-focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    leftborder =cfoc
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
 ==>
   !eval! (set-focus =focus *modelfocus*)
 =goal>
    step set-second-focus
    focus =focus
 =imaginal>
 =retrieval>
)

;; Discontinuous: A unified model that has to be merged was found. Now start the merging process with setting the foci to the leftborders of the two models.
(p p2-search-for-tbmmodel-success-set-second-focus
 =goal>
    ; isa reasoning-task
    phase process-p2
    step set-second-focus
    second-focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    leftborder =cfoc
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
 ==>
   !eval! (set-focus =cfoc *conclusionfocus*)
 =goal>
    step merge-models
    direction right
    second-focus =cfoc
 =imaginal>
 =retrieval>
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Find the term present in both current unified model and tbmmodel (tbmmodel is also a unified model)
;;;
;;; Set both foci (of current and tbm model) to the leftborders.
;;; Now compare the outer left term of the tbmmodel  with all terms of the current unified model, starting with the term at the leftborder. Comparisons proceed to the right.
;;; If all terms of the current model are compared with the focussed term of the tbmmodel, the next term of the tbmmodel is searched in the current unified model by resetting
;;; the focus to the outer left term of the current unified model first/again.  If all terms of the tbmmodel are compared with the terms in the unified model, and no match was
;;; found, both models cannot be merged.
;; Discontinuous: If the focus of the current unified model is NOT at the right border and the focussed terms do not match, the focus of the current unified model will be moved
;; to the next term to the right.
(p p2-compare-terms-focus-not-at-right-border-of-current-model-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step merge-models
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - =foc =val
    - rightborder =foc
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
 =imaginal>
)

;; Discontinuous: If the focus of the current unified model IS at the right border, the focus of the to be merged model is NOT at the right border and the focussed terms do
;; not match, the focus of the current unified model will be set back to the leftmost term and the focus of the to be merged model is moved to the next term to the right.
(p p2-compare-terms-focus-at-right-border-of-current-model-tbmfocus-not-at-right-border-of-tbmmodel-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step merge-models
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - rightborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - =foc =val
    rightborder =foc
    leftborder =lb
 ==>
   !bind! =dummy (set-focus =lb *modelfocus*)
 =goal>
    focus =lb
    step set-moved-focus
 =retrieval>
 =imaginal>
)

;; Discontinuous: If the focus of the current unified model IS at the right border, the focus of the to be merged model is NOT at the right border and the focussed terms do
;; not match, the focus of the current unified model will be set back to the leftmost term and the focus of the to be merged model is moved to the next term to the right.
(p p2-compare-terms-focus-at-right-border-of-current-model-tbmfocus-not-at-right-border-of-tbmmodel-no-match-set-moved-focus
 =goal>
    ; isa reasoning-task
    phase process-p2
    step set-moved-focus
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - rightborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ==>
   !bind! =newcfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    step merge-models
    second-focus =newcfocus
 =retrieval>
 =imaginal>
)

;; Discontinuous: If both foci (of the current unified model and the tbmmodel) and the focussed terms do not match, the compare process is terminated. Continue with next premise.
(p p2-compare-terms-foci-at-right-borders-of-both-current-and-tbmmodel-no-match-terminate-merge-process
 =goal>
    ; isa reasoning-task
    phase process-p2
    step merge-models
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    rightborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - =foc =val
    rightborder =foc
    leftborder =lb
 ==>
=goal>
    ;focus =lb
    second-focus nil
    step prepare-wait-for-first-term
 =imaginal>
)

;; Discontinuous: If both focussed terms(of the current unified model and the tbmmodel)  match, the compare process is terminated and the merging process starts.
(p p2-compare-terms-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step merge-models
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =foc =val
 ==>
=goal>
    direction stop
    step insert-terms
    searchterm =val
 =imaginal>
 =retrieval>
)

;;; Productions for merging the two models after a shared term was found.

;; Discontinuous: The common term in both models were found, this common term is at the right border of the tbm model,
;; so the other terms of the tbm model are inserted to the left of the common term.
(p p2-merge-models-reference-object-right-set-direction-left
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction stop
    second-focus =cfoc
    focus =foc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    rightborder =cfoc
 ==>
   !bind! =newcfocus (first (move-focus-left *conclusionfocus*))
 =goal>
    direction left
    second-focus =newcfocus
 =retrieval>
)

;; Discontinuous: The common term in both models were found, this common term is at the left border of the tbm model,
;; so the other terms of the tbm model are inserted to the right of the common term.
(p p2-merge-models-reference-object-left-set-direction-right
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction stop
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    leftborder =cfoc
 ==>
   !bind! =newcfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    direction right
    second-focus =newcfocus
 =retrieval>
)


;; Discontinuous: The common term in both models were found, this common term is at the left border of the tbm model,
;; so the other terms of the tbm model are inserted to the right of the common term.
(p p2-merge-models-reference-object-middle-set-direction-right
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction stop
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    - leftborder =cfoc
    - rightborder =cfoc
 ==>
   !bind! =newcfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    direction right
    second-focus =newcfocus
 =retrieval>
)

;;;;;;;;;;;;;;; LEFT

;; Discontinuous: A term of the tbm model is inserted in the unified model, continue with the next term to the left.
(p p2-merge-models-move-left-insert-term-focus-not-at-left-border-of-tbm
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction left
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    - leftborder =cfoc
    =cfoc =val
    type nil
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newlb (extend-left =foc)
    !bind! =newfocus (first (move-focus-left *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    step set-second-moved-focus
    focus =newlb
 =imaginal>
    leftborder =newlb
    =newlb =val
    leftborderterm =val
    modelsize =newsize
    =val yes
 =retrieval>
)

;; Discontinuous: A term of the tbm model is inserted in the unified model, continue with the next term to the left.
(p p2-merge-models-move-left-insert-term-focus-not-at-left-border-of-tbm-set-second-moved-focus
 =goal>
    ; isa reasoning-task
    phase process-p2
    step set-second-moved-focus
    direction left
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    - leftborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
 ==>
   !bind! =newcfocus (first (move-focus-left *conclusionfocus*))
 =goal>
    step insert-terms
    second-focus =newcfocus
 =imaginal>
 =retrieval>
)

;; Discontinuous: When the currently focussed position is not empty, move right to the next empty position for further insertions.
(p p2-merge-models-move-left-no-insert-term-focus-at-left-border-of-tbm-move-left
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction left
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    ;leftborder =cfoc
    ;=cfoc =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - leftborder =foc
    - =foc nil
 ==>
   !bind! =newfocus (first (move-focus-left *modelfocus*))
 =goal>
    focus =newfocus
 =imaginal>
 =retrieval>
    type initial
)

;; Discontinuous: A term of the tbm model is inserted in the unified model. There are no further terms,
;; so the merging process will be terminated. Continue with next premise.
(p p2-merge-models-move-left-insert-term-focus-at-left-border-of-tbm
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction left
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    leftborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newlb (extend-left =foc)
    !bind! =newfocus (first (move-focus-left *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    focus =newlb
    second-focus nil
    step prepare-wait-for-first-term
    direction stop
    continuous yes
 =imaginal>
    leftborder =newlb
    =newlb =val
    leftborderterm =val
    modelsize =newsize
    =val yes
)

;; Discontinuous: When the rightborder term of the tbm model is identical to the currently focussed term
;; in the unified model, terminate the merging process.
(p p2-merge-models-move-left-no-insert-term-focus-at-left-border-of-tbm-terminate
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction left
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    leftborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =foc
    =foc =val
 ==>
=goal>
    second-focus nil
    step prepare-wait-for-first-term
    continuous yes
    direction stop
 =imaginal>
)

;;; annotated procedures
;; Discontinuous: A term of the tbm model is inserted in the unified model, continue with the next term to the left.
(p p2-merge-models-move-left-insert-term-focus-not-at-left-border-of-tbm-create-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction left
    focus =foc
    second-focus =cfoc
    searchterm =st
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    - leftborder =cfoc
    =cfoc =val
    type initial
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newlb (extend-left =foc)
    !bind! =newfocus (first (move-focus-left *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    step set-second-moved-focus
    focus =newlb
 =imaginal>
    leftborder =newlb
    =newlb =val
    leftborderterm =val
    modelsize =newsize
    =val yes
 =retrieval>
    modeltype annotated-premise
    loco =val
    refo =st
    reference-term-pos right
    ;;trial =trial ; Restrict to current trial
 +retrieval> =retrieval
)


;; Discontinuous: A term of the tbm model is inserted in the unified model, continue with the next term to the left.
(p p2-merge-models-move-left-insert-term-focus-not-at-left-border-of-tbm-set-second-moved-focus-annotated-premise
 =goal>
    ; isa reasoning-task
    phase process-p2
    step set-second-moved-focus
    direction left
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    tbm yes
    - leftborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
 ==>
   !bind! =newcfocus (first (move-focus-left *conclusionfocus*))
 =goal>
    step insert-terms
    second-focus =newcfocus
 =imaginal>
 =retrieval>
)

;; Discontinuous: A term of the tbm model is inserted in the unified model. There are no further terms,
;; so the merging process will be terminated. Continue with next premise.
(p p2-merge-models-move-left-insert-term-focus-at-left-border-of-tbm-annotated-premise
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction left
    focus =foc
    second-focus =cfoc
    searchterm =st
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    tbm yes
    leftborder =cfoc
    =cfoc =val
    loco =newrefo
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newlb (extend-left =foc)
    !bind! =newfocus (first (move-focus-left *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    focus =newlb
    second-focus nil
    step prepare-wait-for-first-term
    continuous yes
    direction stop
 =imaginal>
    leftborder =newlb
    =newlb =val
    leftborderterm =val
    modelsize =newsize
    =val yes
 =retrieval>
    modeltype annotated-premise
    loco =val
    refo =newrefo
    reference-term-pos right
    ;;trial =trial ; Restrict to current trial
    type inherited
 -retrieval>
)

;; Discontinuous: When the rightborder term of the tbm model is identical to the currently focussed term
;; in the unified model, terminate the merging process.
(p p2-merge-models-move-left-no-insert-term-focus-at-left-border-of-tbm-terminate-annotated-premise
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction left
    focus =foc
    second-focus =cfoc
    searchterm =st
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    tbm yes
    leftborder =cfoc
    =cfoc =val
    loco =newrefo
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =foc
    =foc =val
 ==>
=goal>
    second-focus nil
    step prepare-wait-for-first-term
    continuous yes
    direction stop
 =imaginal>
 =retrieval>
    modeltype annotated-premise
    loco =val
    refo =newrefo
    reference-term-pos right
    ;;trial =trial ; Restrict to current trial
    type inherited
 -retrieval>
)

;;;;;;;;;;;;;;; RIGHT

;; Discontinuous: A term of the tbm model is inserted in the unified model, continue with the next term to the left.
(p p2-merge-models-move-right-insert-term-focus-not-at-right-border-of-tbm
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    - rightborder =cfoc
    =cfoc =val
    type nil
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newrb (extend-right =foc)
    !bind! =newfocus (first (move-focus-right *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    step set-second-moved-focus
    focus =newrb
 =imaginal>
    rightborder =newrb
    =newrb =val
    rightborderterm =val
    modelsize =newsize
    =val yes
 =retrieval>
)

;; Discontinuous: A term of the tbm model is inserted in the unified model, continue with the next term to the left.
(p p2-merge-models-move-right-insert-term-focus-not-at-right-border-of-tbm-set-second-moved-focus
 =goal>
    ; isa reasoning-task
    phase process-p2
    step set-second-moved-focus
    direction right
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    - rightborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
 ==>
   !bind! =newcfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    step insert-terms
    second-focus =newcfocus
 =imaginal>
 =retrieval>
)

;; Discontinuous: When the currently focussed position is not empty, move right to the next empty position for further insertions.
(p p2-merge-models-move-right-no-insert-term-focus-at-right-border-of-tbm-move-right
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    ;rightborder =cfoc ;abc
    ;=cfoc =val ;abc
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - rightborder =foc
    - =foc nil
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =imaginal>
 =retrieval>
    type initial
)

;; Discontinuous: A term of the tbm model is inserted in the unified model. There are no further terms,
;; so the merging process will be terminated. Continue with next premise.
(p p2-merge-models-move-right-insert-term-focus-at-right-border-of-tbm
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    rightborder =cfoc
    =cfoc =val
    leftborderterm =st
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newlb (extend-right =foc)
    !bind! =newfocus (first (move-focus-right *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    focus =newlb
    second-focus nil
    step prepare-wait-for-first-term
    continuous yes
    direction stop
 =imaginal>
    rightborder =newlb
    =newlb =val
    rightborderterm =val
    modelsize =newsize
    =val yes
)

;; Discontinuous: When the rightborder term of the tbm model is identical to the currently focussed term
;; in the unified model, terminate the merging process.
(p p2-merge-models-move-right-no-insert-term-focus-at-right-border-of-tbm-terminate
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    rightborder =cfoc
    =cfoc =val
    leftborderterm =st
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    =foc =val
 ==>
=goal>
    second-focus nil
    step prepare-wait-for-first-term
    continuous yes
    direction stop
 =imaginal>
)

;;; annotated procedures
;; Discontinuous: A term of the tbm model is inserted in the unified model, continue with the next term to the left.
(p p2-merge-models-move-right-insert-term-focus-not-at-right-border-of-tbm-create-annotation
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    - rightborder =cfoc
    =cfoc =val
    type initial
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newrb (extend-right =foc)
    !bind! =newfocus (first (move-focus-right *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    step set-second-moved-focus
    focus =newrb
 =imaginal>
    rightborder =newrb
    =newrb =val
    rightborderterm =val
    modelsize =newsize
    =val yes
 =retrieval>
    modeltype annotated-premise
    loco =val
    refo =st
    reference-term-pos left
    ;;trial =trial ; Restrict to current trial
 +retrieval> =retrieval
)


;; Discontinuous: A term of the tbm model is inserted in the unified model, continue with the next term to the left.
(p p2-merge-models-move-right-insert-term-focus-not-at-right-border-of-tbm-set-second-moved-focus-annotated-premise
 =goal>
    ; isa reasoning-task
    phase process-p2
    step set-second-moved-focus
    direction right
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    tbm yes
    - rightborder =cfoc
    =cfoc =val
 =imaginal>
    ; isa mentalmodel
 ==>
   !bind! =newcfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    step insert-terms
    second-focus =newcfocus
 =imaginal>
 =retrieval>
)

;; Discontinuous: A term of the tbm model is inserted in the unified model. There are no further terms,
;; so the merging process will be terminated. Continue with next premise.
(p p2-merge-models-move-right-insert-term-focus-at-right-border-of-tbm-annotated-premise
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    tbm yes
    rightborder =cfoc
    =cfoc =val
    loco =newrefo
    leftborderterm =st
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newrb (extend-right =foc)
    !bind! =newfocus (first (move-focus-right *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    focus =newrb
    second-focus nil
    step prepare-wait-for-first-term
    continuous yes
    direction stop
 =imaginal>
    rightborder =newrb
    =newrb =val
    rightborderterm =val
    modelsize =newsize
    =val yes
 =retrieval>
    modeltype annotated-premise
    loco =val
    refo =newrefo
    reference-term-pos left
    ;;trial =trial ; Restrict to current trial
    type inherited
 -retrieval>
)

;; Discontinuous: When the rightborder term of the tbm model is identical to the currently focussed term
;; in the unified model, terminate the merging process.
(p p2-merge-models-move-right-no-insert-term-focus-at-right-border-of-tbm-terminate-annotated-premise
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    tbm yes
    rightborder =cfoc
    =cfoc =val
    loco =newrefo
    leftborderterm =st
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    =foc =val
 ==>
=goal>
    second-focus nil
    step prepare-wait-for-first-term
    continuous yes
    direction stop
 =imaginal>
 =retrieval>
    modeltype annotated-premise
    loco =val
    refo =newrefo
    reference-term-pos left
    ;;trial =trial ; Restrict to current trial
    type inherited
 -retrieval>
)

;; productions for now running left

;; Discontinuous: A term of the tbm model is inserted in the unified model. There are no further terms,
;; so the merging process will be terminated. Continue with next premise.
(p p2-merge-models-move-right-insert-term-focus-at-right-border-of-tbm-continue-with-terms-left-of-common-term
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    rightborder =cfoc
    =cfoc =val
    - leftborderterm =st
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newlb (extend-right =foc)
    !bind! =newfocus (first (move-focus-right *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    focus nil
    second-focus nil
    step reset-foci
    continuous yes
    direction stop
 =imaginal>
    rightborder =newlb
    =newlb =val
    rightborderterm =val
    modelsize =newsize
    =val yes
 =retrieval>
)

;; Discontinuous: When the rightborder term of the tbm model is identical to the currently focussed term
;; in the unified model, terminate the merging process.
(p p2-merge-models-move-right-no-insert-term-focus-at-right-border-of-tbm-terminate-continue-with-terms-left-of-common-term
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    tbm yes
    rightborder =cfoc
    =cfoc =val
    - leftborderterm =st
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    =foc =val
 ==>
=goal>
 focus nil
    second-focus nil
    step reset-foci
    continuous yes
    direction stop
 =imaginal>
 =retrieval>
)

;; Discontinuous: A term of the tbm model is inserted in the unified model. There are no further terms,
;; so the merging process will be terminated. Continue with next premise.
(p p2-merge-models-move-right-insert-term-focus-at-right-border-of-tbm-annotated-premise-continue-with-terms-left-of-common-term
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    tbm yes
    rightborder =cfoc
    =cfoc =val
    loco =newrefo
    - leftborderterm =st
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    - =foc nil
    modelsize =size
 ==>
   !bind! =newrb (extend-right =foc)
    !bind! =newfocus (first (move-focus-right *modelfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    focus nil
    second-focus nil
    step reset-foci
    continuous yes
    direction stop
 =imaginal>
    rightborder =newrb
    =newrb =val
    rightborderterm =val
    modelsize =newsize
    =val yes
 =retrieval>
    modeltype annotated-premise
    loco =val
    refo =newrefo
    reference-term-pos left
    ;;trial =trial ; Restrict to current trial
    type inherited
 +retrieval> =retrieval
)

;; Discontinuous: When the rightborder term of the tbm model is identical to the currently focussed term
;; in the unified model, terminate the merging process.
(p p2-merge-models-move-right-no-insert-term-focus-at-right-border-of-tbm-terminate-annotated-premise-continue-with-terms-left-of-common-term
 =goal>
    ; isa reasoning-task
    phase process-p2
    step insert-terms
    direction right
    focus =foc
    second-focus =cfoc
    searchterm =st
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    tbm yes
    rightborder =cfoc
    =cfoc =val
    loco =newrefo
    - leftborderterm =st
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =foc
    =foc =val
 ==>
=goal>
    focus nil
    second-focus nil
    step reset-foci
    continuous yes
    direction stop
 =imaginal>
 =retrieval>
    modeltype annotated-premise
    loco =val
    refo =newrefo
    reference-term-pos left
    ;;trial =trial ; Restrict to current trial
    type inherited
 +retrieval> =retrieval
)

; When the common term is not at the left or right border, and after all terms right of the common term were inserted into the unified mental model, start with inserting the
; terms left of the common term by first setting the focus back to the leftborder of the unified model
(p p2-merge-models-reset-foci-set-first-focus
 =goal>
    ; isa reasoning-task
    phase process-p2
    step reset-foci
    focus nil
    second-focus nil
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
 =retrieval>
    ; isa mentalmodel
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    focus =lb
 =imaginal>
 =retrieval>
    modeltype unified
    loco nil
    refo nil
    reference-term-pos nil
    trial nil
    type nil
)

; ...and continue searching for the common term...
(p p2-merge-models-reset-foci-find-first-focus-position-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step reset-foci
    focus =f
    second-focus nil
    searchterm =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - =f =val
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =imaginal>
)

; ...until it is found again in the unified model. Now start searching for the common term in the tbm model.
(p p2-merge-models-reset-foci-find-first-focus-position-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step reset-foci
    focus =f
    second-focus nil
    searchterm =val
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =f =val
 =retrieval>
    ; isa mentalmodel
    leftborder =lb
 ==>
   !eval! (set-focus =lb *conclusionfocus*)
 =goal>
    second-focus =lb
 =imaginal>
 =retrieval>
)

; ...and continue searching for the common term...
(p p2-merge-models-reset-foci-find-second-focus-position-no-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step reset-foci
    - focus nil
    second-focus =f
    searchterm =val
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    - =f =val
 ==>
   !bind! =newfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    second-focus =newfocus
 =retrieval>
)

; ...until it is found again in the tbm model. Now the model uses the same productions as when only terms left of the common term have to be inserted.
(p p2-merge-models-reset-foci-find-second-focus-position-match
 =goal>
    ; isa reasoning-task
    phase process-p2
    step reset-foci
    - focus nil
    second-focus =f
    searchterm =val
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =f =val
 ==>
   !bind! =newcfocus (first (move-focus-left *conclusionfocus*))
 =goal>
    direction left
    second-focus =newcfocus
    step insert-terms
 =retrieval>
)

;;; Production for a/no keypress when paced is set to self/externally



;; Selfpaced: The current premise was successfully processed. Continue with next premise.
(p p2-process-premise-complete
 =goal>
    ; isa reasoning-task
    phase process-p2
    step prepare-wait-for-first-term
 ?imaginal>
    state free
 ==>
  !bind! =time (mod (get-time) 24)
 =goal>
    step wait-for-first-term
    time =time
 +goal> =goal
 +visual>
    isa clear
)

;;; Production for completing the process-p2 phase. Phase is complete if the number of the processed premises reaches the total number of premises of the task.

;; All premises were processed, continue with processing the conclusion.
(p p2-process-premises-complete
 =goal>
    ; isa reasoning-task
    phase process-p2
    step wait-for-first-term
    > number 2
    direction =direction
    trial =trial
    annotations =annotations
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
 ==>
 =goal>
    phase process-c
    direction =direction
    trial =trial
    annotations =annotations
    step nil
    leftborder nil
    rightborder nil
    modelsize nil
 =imaginal>
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; PC: PROCESS CONCLUSION ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Productions for reading the conclusion and store it into a mentalmodel chunk of type 'conclusion.


;; When the first term of the conclusion has been presented, the attention is shifted to this object.
(p pc-encode-first-term
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    leftborder nil
    rightborder nil
    modelsize nil
 =visual-location>
    isa visual-location
 ?visual>
    state free
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 ==>
=goal>
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
    step retrieve-term
    pmmleftborderterm =lbt
    pmmrightborderterm =rbt
 +visual>
    isa move-attention
    screen-pos =visual-location
 =visual-location>
 =imaginal>
    model pmm
)

;; see above: p2-retrieve-first-term, etc.
(p pc-retrieve-first-term
 =goal>
    ; isa reasoning-task
    phase process-c
    step retrieve-term
 =visual>
    isa visual-object
    value =value
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ?retrieval>
    - state busy
 ==>
=goal>
    step nil
 +retrieval>
    ; isa term
    - name nil
    value =value
 =visual>
 =imaginal>
)

;; retrieval of first term was not successful. Try again.
(p pc-retrieve-first-term-retry
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
 =visual>
    isa visual-object
    value =value
 ?retrieval>
    state error
 ==>
=goal>
 +retrieval>
    ; isa term
    - name nil
    value =value
 =visual>
)

;; When the second term of the conclusion has been presented, the imaginal buffer contains a mentalmodel chunk of type 'conclusion.
(p pc-encode-second-term
 =goal>
    ; isa reasoning-task
    phase process-c
    step consecutive-conclusion-processing
 =visual-location>
    isa visual-location
 ?visual>
    state free
    - state busy
    - buffer full
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
 ?retrieval>
    - state busy
 ==>
=goal>
    step encode-term
 +visual>
    isa move-attention
    screen-pos =visual-location
 =visual-location>
 =imaginal>
)

;; see above: p2-retrieve-first-term, etc.
(p pc-retrieve-second-term
 =goal>
    ; isa reasoning-task
    phase process-c
    step encode-term
 =visual>
    isa visual-object
    value =value
 ?retrieval>
    - state busy
 ==>
 =goal>
    step consecutive-conclusion-processing
 +retrieval>
    ; isa term
    - name nil
    value =value
 =visual>
)

;; see above: p2-retrieve-first-term, etc.
(p pc-retrieve-second-term-retry
 =goal>
    ; isa reasoning-task
    phase process-c
    step consecutive-conclusion-processing
 =visual>
    isa visual-object
    value =value
 ?retrieval>
    state error
 ==>
=goal>
 +retrieval>
    ; isa term
    - name nil
    value =value
 =visual>
)

;; Add the second term left to the first one if there is a conclusion chunk in the imaginal buffer
(p pc-insert-second-term-left-recall-unified-model
 =goal>
    ; isa reasoning-task
    phase process-c
    step consecutive-conclusion-processing
    leftborder =lbu
    rightborder =rbu
 =visual-location>
    isa visual-location
    < screen-x 150
 =retrieval>
    ; isa term
    - name nil
    name =name
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =lb
    modelsize =size
 =visual>
    isa visual-object
 ==>
   !bind! =nlb (extend-left =lb)
    !bind! =newfocus (first (move-focus-left *conclusionfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    step nil
 =imaginal>
    leftborder =nlb
    =newfocus =name
    leftborderterm =name
    modelsize =newsize
    =name yes
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lbu
    rightborder =rbu
    :recently-retrieved t
    model pmm
)

;; There is already a conclusion chunk in the imaginal buffer, so just add the second term right to the first one.
(p pc-insert-second-term-right-recall-unified-model
 =goal>
    ; isa reasoning-task
    phase process-c
    step consecutive-conclusion-processing
    leftborder =lbu
    rightborder =rbu
 =visual-location>
    isa visual-location
    > screen-x 150
 =retrieval>
    ; isa term
    - name nil
    name =name
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    rightborder =rb
    modelsize =size
 =visual>
    isa visual-object
 ==>
   !bind! =nrb (extend-right =rb)
    !bind! =newfocus (first (move-focus-right *conclusionfocus*))
    !bind! =newsize (+ =size 1)
 =goal>
    step nil
 =imaginal>
    rightborder =nrb
    =newfocus =name
    rightborderterm =name
    modelsize =newsize
    =name yes
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lbu
    rightborder =rbu
    :recently-retrieved t
    model pmm
)

;; Retrieval of unified model was not successful. Try again.
(p pc-insert-second-term-recall-unified-model-retry
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    leftborder =lbu
    rightborder =rbu
 ?retrieval>
    - state busy
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
 ==>
=goal>
    step nil
 =imaginal>
 +retrieval>

    modeltype unified
    tbm nil
    leftborder =lbu
    rightborder =rbu
    model pmm
)

;; When the imaginal buffer is cleared, the unified mentalmodel must be retrieved in order to later  compare it with the conclusion.
(p pc-create-conclusion
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    intermediateretrieval no
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 =retrieval>
    ; isa term
    - name nil
    name =name
 =visual>
    isa visual-object
 ==>
=goal>
 +imaginal>
    isa mentalmodel
    modeltype conclusion
    pos1 =name
    leftborderterm =name
    rightborderterm =name
    =name yes
)

;; When the imaginal buffer is cleared, the unified mentalmodel must be retrieved in order to later  compare it with the conclusion.
(p pc-create-conclusion-intermediate-retrieval2
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    intermediateretrieval yes
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 =retrieval>
    ; isa term
    - name nil
    name =name
 =visual>
    isa visual-object
 =visual-location>
    isa visual-location
 ==>
=goal>
 +imaginal>
    isa mentalmodel
    modeltype conclusion
    pos1 =name
    leftborderterm =name
    rightborderterm =name
    =name yes
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
    model pmm
   =visual-location>

)

;; When the imaginal buffer is cleared, the unified mentalmodel must be retrieved in order to later  compare it with the conclusion.
(p pc-create-conclusion-intermediate-retrieval-retry
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    intermediateretrieval yes
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
 ?retrieval>
    state error
 ==>
=goal>
 =imaginal>
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
    model pmm
)

;; Check if there are more terms presented
(p pc-check-for-more-terms-on-the-screen
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    intermediateretrieval no
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =lbc
 =visual-location>
    isa visual-location
    screen-x =x
    screen-y =y
 ?retrieval>
    state free
    - buffer full
 ==>
   !eval! (set-focus =lbc *conclusionfocus*)
 =goal>
    step search-for-more-conclusion-terms
    second-focus =lbc
 =imaginal>
 +visual-location>
    isa visual-location
    :attended nil
    > screen-x =x
    screen-y =y
    screen-x lowest
)

;; Check if there are more terms presented
(p pc-check-for-more-terms-on-the-screen-intermediate-retrieval
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    intermediateretrieval yes
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =lbc
 =visual-location>
    isa visual-location
    screen-x =x
    screen-y =y
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 ==>
   !eval! (set-focus =lbc *conclusionfocus*)
 =goal>
    step search-for-more-conclusion-terms
    second-focus =lbc
 =imaginal>
 =retrieval>
 +visual-location>
    isa visual-location
    :attended nil
    > screen-x =x
    screen-y =y
    screen-x lowest
)



;; This the first production representing the junction for the two presentation types: either  terms are presented consecutively, or a complete mental model is presented.
;; In this case there is no further term and therefore the model needs to wait for the next term to be presented.
(p pc-check-for-more-terms-failure-wait-for-second-term
   !bind! =time (mod (get-time) 24)
 =goal>
    ; isa reasoning-task
    phase process-c
    step search-for-more-conclusion-terms
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
 ?visual-location>
    state error
 ==>
 =goal>
    step consecutive-conclusion-processing
    time =time
 =imaginal>
 +visual-location>
    isa visual-location
    :attended nil
)




(p pc-process-p2-took-way-to-long-case-2-roooobin
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    leftborder nil
    rightborder nil
    modelsize nil
    > time 12000
 ?visual-location>
    buffer empty
 ?visual>
    state free
    - state busy
    - buffer full
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 ==>
=goal>
     step kasjdaskjd
)






(p pc-check-no-more-conclusion-so-first-one-took-to-long-start-guessing-roooobin
 =goal>
    ; isa reasoning-task
    phase process-c
    > time 12000
    step consecutive-conclusion-processing
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
 ?visual-location>
    state error
    buffer failure
  ?visual>
    state free
    - state busy
    - buffer full
 ?retrieval>
    - state busy

 ==>
=goal>
    step whabbadabbua
 =imaginal>
)




;; This the second production representing the junction for the two presentation types: either terms are presented consecutively, or a complete mental model is presented.
;; In this case the next step would be to retrieve the respective term.
(p pc-check-for-more-terms-success
 =goal>
    ; isa reasoning-task
    phase process-c
    step search-for-more-conclusion-terms
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    rightborder =rb
 =visual-location>
    isa visual-location
 ==>

    !bind! =nrb (extend-right =rb)
    !bind! =newfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    step model-conclusion-processing
    focus =newfocus
 =imaginal>
    rightborder =nrb
    =nrb nil
 +imaginal> =imaginal
 =visual-location>
 +visual>
    isa move-attention
    screen-pos =visual-location
)

;; When the textual item representing a term has been presented/seen, the respective chunk of type 'term is retrieved.
(p pc-retrieve-next-term
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
    intermediateretrieval no
 =visual>
    isa visual-object
    value =value
 ?retrieval>
    - state busy
 ==>
=goal>
 +retrieval>
    ; isa term
    - name nil
    value =value
)

;; When the textual item representing a term has been presented/seen, the respective chunk of type 'term is retrieved.
(p pc-retrieve-next-term-intermediate-retrieval
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
    intermediateretrieval yes
 =visual>
    isa visual-object
    value =value
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
 ==>
=goal>
    leftborder =lb
    rightborder =rb
    searchterm =value
 +retrieval>
    ; isa term
    - name nil
    value =value
)

;; When the textual item representing a term has been presented/seen, the respective chunk of type 'term is retrieved.
(p pc-retrieve-next-term-intermediate-retrieval-retry
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
    intermediateretrieval yes
    searchterm =value
 ?retrieval>
    state error
 ==>
=goal>
 +retrieval>
    ; isa term
    - name nil
    value =value
)

;; Integrate retrieved term into the mental model of type 'conclusion
(p pc-insert-next-term-and-check-for-more-terms
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
    focus =focus
    intermediateretrieval no
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    =focus nil
    modelsize =size
 =visual-location>
    isa visual-location
    screen-x =x
 =retrieval>
    ; isa term
    - name nil
    name =name
 ==>
   !bind! =newsize (+ =size 1)
 =goal>
    step model-conclusion-processing
 =imaginal>
    =focus =name
    rightborderterm =name
    modelsize =newsize
    =name yes
 +visual-location>
    isa visual-location
    :attended nil
    > screen-x =x
    screen-x lowest
)

;; Integrate retrieved term into the mental model of type 'conclusion
(p pc-insert-next-term-and-check-for-more-terms-intermediate-retrieval
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
    focus =focus
    leftborder =lb
    rightborder =rb
    intermediateretrieval yes
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    =focus nil
    modelsize =size
 =visual-location>
    isa visual-location
    screen-x =x
    screen-y =y
 =retrieval>
    ; isa term
    - name nil
    name =name
 ==>
   !bind! =newsize (+ =size 1)
 =goal>
    searchterm nil
 =imaginal>
    =focus =name
    rightborderterm =name
    modelsize =newsize
    =name yes
 +visual-location>
    isa visual-location
    :attended nil
    > screen-x =x
    screen-y =y
    screen-x lowest
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
    model pmm
)

;; Integrate retrieved term into the mental model of type 'conclusion
(p pc-insert-next-term-and-check-for-more-terms-intermediate-retrieval-retry
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
    focus =focus
    leftborder =lb
    rightborder =rb
    intermediateretrieval yes
    searchterm nil
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    =focus nil
    modelsize =size
 =visual-location>
    isa visual-location
    screen-x =x
    screen-y =y
 ?retrieval>
    state error
 ==>
   !bind! =newsize (+ =size 1)
 =goal>
 =imaginal>
 =visual-location>
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lb
    rightborder =rb
    model pmm
)

;; Proceed with checking for more terms.
(p pc-check-for-more-terms-on-the-screen-success-proceed
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
    intermediateretrieval no
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    rightborder =rb
 =visual-location>
    isa visual-location
 ?visual>
    buffer empty
    - state busy
 ?retrieval>
    buffer empty
    - state busy
 ==>
   !bind! =nrb (extend-right =rb)
    !bind! =newfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    step model-conclusion-processing
    focus =newfocus
 =imaginal>
    rightborder =nrb
    =nrb nil
 +imaginal> =imaginal
 =visual-location>
 +visual>
    isa move-attention
    screen-pos =visual-location
)

;; Proceed with checking for more terms.
(p pc-check-for-more-terms-on-the-screen-success-proceed-intermediate-retrieval
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
    intermediateretrieval yes
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    rightborder =rb
 =visual-location>
    isa visual-location
 ?visual>
    buffer empty
    - state busy
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 ==>
   !bind! =nrb (extend-right =rb)
    !bind! =newfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    step model-conclusion-processing
    focus =newfocus
 =imaginal>
    rightborder =nrb
    =nrb nil
 +imaginal> =imaginal
 =visual-location>
 +visual>
    isa move-attention
    screen-pos =visual-location
 =retrieval>
)

;; There is no term left that can be processed. Terminate conclusion / model processing phase.
(p pc-check-for-more-terms-failure-all-terms-seen
 =goal>
    ; isa reasoning-task
    phase process-c
    step model-conclusion-processing
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
 ?visual-location>
    state error
 ==>
=goal>
    step nil
 =imaginal>
)



;; conclusion chunk is complete, proceed with step 'compare-model-with-conclusion.
(p pc-complete-no-space
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =lb
    - rightborder =lb
    modelsize =cs
 ?retrieval>
    - buffer empty
    - state busy
    - state error
 ==>
 =goal>
    phase inspection
    step compare-model-with-conclusion
    searchterm nil
    focus nil
    second-focus nil
    direction stop
    conclusionsize =cs
 +goal> =goal
 =imaginal>
)



;; conclusion chunk is complete, proceed with step 'compare-model-with-conclusion.
(p pc-complete-retrieve-unified-model-no-space
 =goal>
    ; isa reasoning-task
    phase process-c
    step nil
    leftborder =lbm
    rightborder =rbm
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =lb
    - rightborder =lb
    rightborder =rb
    modelsize =cs
    =lb =v1
    =rb =v2
 ?retrieval>
    buffer empty
    - state busy
 ==>
=goal>
    phase inspection
    step compare-model-with-conclusion
    searchterm nil
    focus nil
    second-focus nil
    direction stop
    conclusionsize =cs
 +goal> =goal
 =imaginal>
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lbm
    rightborder =rbm
    model pmm
)

;; conclusion chunk is complete, proceed with step 'compare-model-with-conclusion.
(p pc-complete-retrieve-unified-model-retry
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    leftborder =lbm
    rightborder =rbm
    variation nil
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =lb
    - rightborder =lb
    modelsize =cs
 ?retrieval>
    state error
 ==>
=goal>
 =imaginal>
 +retrieval>
    modeltype unified
    tbm nil
    leftborder =lbm
    rightborder =rbm
    model pmm
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; I: INSPECT ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Productions for comparing the conclusion in the imaginal with the unified mentalmodel in the retrieval buffer.

;; 1) Go from left to right through the mental model and first search for the leftmost term.
;; 2) Continue going right for finding the next terms of the conclusion
;; 3) When all conclusion terms are found in the correct order: press R
;; 4) When the term is not found in the correct order, i.e. when the right border of the mental model is reached.

(p i-presearch-if-borderterms-are-identical-yes
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction stop
    focus nil
    second-focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
    leftborderterm =lbt
    rightborderterm =rbt
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =cfoc
    leftborderterm =lbt
    rightborderterm =rbt
 ?visual-location>
    - buffer full
 ==>
=goal>
    phase respond
    step respond-with-r
 +imaginal> =retrieval
)


(p i-presearch-if-borderterms-are-identical-no
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction stop
    focus nil
    second-focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
    leftborderterm =lbt
    rightborderterm =rbt
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =cfoc
    leftborderterm =rbt
    rightborderterm =lbt
 ?visual-location>
    - buffer full
 ==>
=goal>
    phase respond
    step respond-with-f
 +imaginal> =retrieval
)

(p i-presearch-if-borderterms-are-identical-left-term-is-no-conclusion-term
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction stop
    focus nil
    second-focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
    leftborderterm =lbt
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =cfoc
    - leftborderterm =lbt
    - rightborderterm =lbt
 ?visual-location>
    - buffer full
 ==>
   !eval! (set-focus =focus *modelfocus*)
 =goal>
    focus =focus
    step compare-model-with-conclusion
 =retrieval>
 =imaginal>
)

(p i-presearch-if-borderterms-are-identical-right-term-is-no-conclusion-term
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction stop
    focus nil
    second-focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
    rightborderterm =rbt
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =cfoc
    - leftborderterm =rbt
    - rightborderterm =rbt
 ?visual-location>
   ; changed...
    - buffer full
 ==>
   !eval! (set-focus =focus *modelfocus*)
 =goal>
    focus =focus
    step compare-model-with-conclusion
 =retrieval>
 =imaginal>
)

;; Set the focus to the most left positions of both the unified model and the conclusion.
(p i-set-direction-right-first-focus
 =goal>
    ; isa reasoning-task
    phase inspection
    step presearch-done
    direction stop
    focus nil
    second-focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =cfoc
 ?visual-location>
    ;changed
    - buffer full
 ==>
   !eval! (set-focus =focus *modelfocus*)
 =goal>
    focus =focus
    step compare-model-with-conclusion
 =retrieval>
 =imaginal>
)

;; Set the focus to the most left positions of both the unified model and the conclusion.
(p i-set-direction-right-second-focus
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction stop
    focus =f
    second-focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =focus
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    leftborder =cfoc
 ==>
   !eval! (set-focus =cfoc *conclusionfocus*)
 =goal>
    direction right
    second-focus =cfoc
 =retrieval>
 =imaginal>
)

; Mental model NOT at rightborder
; Conclusion NOT at rightborder
; Match
(p i-move-right-foci-not-at-right-borders-of-both-conclusion-and-current-model-match-set-first-focus
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction right
    focus =focus
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =focus =value
    - rightborder =focus
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    =cfoc =value
    - rightborder =cfoc
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    step set-second-focus
    focus =newfocus
 =retrieval>
 =imaginal>
)

; Mental model NOT at rightborder
; Conclusion NOT at rightborder
; Match
(p i-move-right-foci-not-at-right-borders-of-both-conclusion-and-current-model-match-set-second-focus
 =goal>
    ; isa reasoning-task
    phase inspection
    step set-second-focus
    direction right
    focus =focus
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    =cfoc =value
    - rightborder =cfoc
 ==>
   !bind! =newcfocus (first (move-focus-right *conclusionfocus*))
 =goal>
    step compare-model-with-conclusion
    second-focus =newcfocus
 =retrieval>
 =imaginal>
)

; Mental model at rightborder
; Conclusion NOT at rightborder
; Match
(p i-move-right-cfocus-not-at-right-border-of-conclusion-but-focus-at-right-border-of-current-model-match
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction right
    focus =focus
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =focus =value
    rightborder =focus
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    =cfoc =value
    - rightborder =cfoc
    leftborderterm =lbt
    rightborderterm =rbt
 ==>
=goal>
    direction stop
    step check-for-initial-annotation
    leftborderterm =lbt
    rightborderterm =rbt
 +imaginal> =retrieval
)
;Mental model somewhere
; Conclusion at rightborder
; Match
(p i-move-right-cfocus-at-right-border-of-conclusion-match-press-r
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction right
    focus =focus
    second-focus =cfoc
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =focus =value
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    =cfoc =value
    rightborder =cfoc
 ==>
=goal>
    phase respond
    step respond-with-r
 +imaginal> =retrieval
)

; Mental model somewhere
; Conclusion not at rightborder
; No match
;; Shortcut production
(p i-move-right-cfocus-not-at-right-border-of-conclusion-no-match-same-size
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction right
    focus =focus
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =focus =value
    modelsize =size
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    - =cfoc =value
    - rightborder =cfoc
    modelsize =size
    leftborderterm =lbt
    rightborderterm =rbt
 ==>
=goal>
    direction stop
    step check-for-initial-annotation
    leftborderterm =lbt
    rightborderterm =rbt
 +imaginal> =retrieval
)

; Mental model somewhere
; Conclusion not at rightborder
; No match
(p i-move-right-cfocus-not-at-right-border-of-conclusion-no-match-not-same-size
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction right
    focus =focus
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =focus =value
    modelsize =size
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    - =cfoc =value
    - rightborder =cfoc
    - modelsize =size
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
 =imaginal>
)

; Mental model somewhere
; Conclusion at rightborder
; No match
(p i-move-right-cfocus-at-right-border-of-conclusion-no-match
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction right
    focus =focus
    second-focus =cfoc
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =focus =value
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    - =cfoc =value
    rightborder =cfoc
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
 =imaginal>
)

; Mental model at rightborder
; Conclusion somewhere
; No match
(p i-move-right-focus-at-right-border-of-current-model-no-match
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    direction right
    focus =focus
    second-focus =cfoc
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype unified
    =focus =value
    rightborder =focus
 =imaginal>
    ; isa mentalmodel
    modeltype conclusion
    - =cfoc =value
    leftborderterm =lbt
    rightborderterm =rbt
 ==>
=goal>
    direction stop
    step check-for-initial-annotation
    leftborderterm =lbt
    rightborderterm =rbt
 +imaginal> =retrieval
)

;; the conclusion wasn't found in the correct order. Look for annotations if there are uncertain positions for terms.
(p i-press-f
 =goal>
    ; isa reasoning-task
    phase inspection
    step check-for-initial-annotation
    annotations no
    searchterm nil
    trial =trial
 ?retrieval>
    buffer empty
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ==>
=goal>
    phase respond
    step respond-with-f
 =imaginal>
)

;;; The conclusion does not match the unified model but there are annotations. Retrieving an annotated premise initiates the variation phase.
(p i-find-initial-annotation
 =goal>
    ; isa reasoning-task
    phase inspection
    step check-for-initial-annotation
    annotations yes
    searchterm nil
    trial =trial
 ?retrieval>
    buffer empty
    - state busy
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ?visual-location>
    - buffer full
 ==>
=goal>
    phase variation
    step variation
    variation yes ; write the conclusion into the premise slot of the reasoning-task
    annotations yes
    trial =trial
    focus nil
    second-focus nil
 +goal> =goal
 =imaginal>
 +retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    type initial
    :recently-retrieved nil
    ;;trial =trial ; Restrict to current trial
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; V: VARIATION ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Productions for varying the unified mentalmodel according to the premises.


;;; Random choice? The initiation of the variation phase failed because no annotated premise could be remembered/retrieved.
;;; Do participants make a random choice (chance level)? If so, the model needs a second production pressing 'r.
(p v-found-no-initial-annotation-press-f
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    - variation nil
    trial =trial
    searchterm nil
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ?manual>
    state free
 ==>
=goal>
    phase respond
    step respond-with-f
 =imaginal>
)

;;; Random choice? The initiation of the variation phase failed because no annotated premise could be remembered/retrieved.
;;; Do participants make a random choice (chance level)? If so, the model needs a second production pressing 'f.
(p v-found-no-initial-annotation-press-r
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    - variation nil
    trial =trial
    searchterm nil
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ==>
=goal>
    phase respond
    step respond-with-r
 =imaginal>
)

;; Steps of variation:
;; 1) Find an annotation of type initial.
;; 2) Find the term in the model.
;; 3) If swap with other term does not contradict with the annotation relation
;; swap this annotated term nearer to the ro.
;; 4) Check if conclusion is now valid.
;; 4a) If yes, press R.
;; 4b) If not, continue with 5).
;; 5) Find an annotation of type inherited.
;; 6) Find the term in the model.
;; 7) If swap with other term does not contradict with the annotation relation
;; swap this annotated term nearer to the ro.
;; 8) Check if conclusion is now valid.
;; 8a) If yes, press R.
;; 8b) If not, try to retrieve other recently not retrieved annotations.
;; 8ba) No further annotation found, press F.
;; 8bb) Furter annotation found, continue with 5).
;; STEP 2
;; Sets the position of the term that has to be interchanged with the annotated term into the focus.
;; If the loco is the at the leftborder of the annotated premise, we know that we have to move the
;; loco to the right; at first determine the name of the position to the right (see the !bind!s).
(p v-set-focus-to-the-position-of-the-annotated-term-right
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =pos
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =pos =loco
 ==>
   !bind! =dummy (set-focus =pos *modelfocus*)                               ; 1. Set focus to to the position of the loco.
 =goal>
    step set-moved-focus
    direction right
    searchterm nil
 =retrieval>
 =imaginal>
)

;; Sets the position of the term that has to be interchanged with the annotated term into the focus.
;; If the loco is the at the leftborder of the annotated premise, we know that we have to move the
;; loco to the right; at first determine the name of the position to the right (see the !bind!s).
(p v-set-focus-to-the-position-of-the-annotated-term-right-set-moved-focus
 =goal>
    ; isa reasoning-task
    phase variation
    step set-moved-focus
    focus nil
    direction right
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =pos
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =pos =loco
    - rightborder =pos
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*)) ; 2. Move focus to the right.
 =goal>
    step variation
    focus =newfocus
    searchterm nil
 =retrieval>
 =imaginal>
)

;; Sets the position of the term that has to be interchanged with the annotated term into the focus.
;; If the loco is the at the leftborder of the annotated premise, we know that we have to move the
;; loco to the right; at first determine the name of the position to the right (see the !bind!s).
(p v-set-focus-to-the-position-of-the-annotated-term-right-at-right-border
 =goal>
    ; isa reasoning-task
    phase variation
    step set-moved-focus
    focus nil
    direction right
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =pos
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =pos =loco
    rightborder =pos
    leftborder =leftposition
 ==>
   !eval! (set-focus =leftposition *modelfocus*)
 =goal>
    step check-for-initial-annotation
    direction stop
    searchterm nil
    phase inspection
 =imaginal>
)

;; Sets the position of the term that has to be interchanged with the annotated term into the focus.
;; If the loco is the at the rightborder of the annotated premise, we know that we have to move the
;; loco to the left; at first determine the name of the position to the left (see the !bind!s).
(p v-set-focus-to-the-position-of-the-annotated-term-left
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =pos
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =pos =loco
 ==>
   !bind! =dummy (set-focus =pos *modelfocus*)                              ; 1. Set focus to to the position of the loco.
 =goal>
    step set-moved-focus
    direction left
    searchterm nil
 =retrieval>
 =imaginal>
)

;; Sets the position of the term that has to be interchanged with the annotated term into the focus.
;; If the loco is the at the rightborder of the annotated premise, we know that we have to move the
;; loco to the left; at first determine the name of the position to the left (see the !bind!s).
(p v-set-focus-to-the-position-of-the-annotated-term-left-set-moved-focus
 =goal>
    ; isa reasoning-task
    phase variation
    step set-moved-focus
    focus nil
    direction left
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =pos
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =pos =loco
    - leftborder =pos
 ==>
   !bind! =newfocus (first (move-focus-left *modelfocus*)) ; 2. Move focus to the left.
 =goal>
    step variation
    focus =newfocus
    searchterm nil
 =retrieval>
 =imaginal>
)

;; Sets the position of the term that has to be interchanged with the annotated term into the focus.
;; If the loco is the at the rightborder of the annotated premise, we know that we have to move the
;; loco to the left; at first determine the name of the position to the left (see the !bind!s).
(p v-set-focus-to-the-position-of-the-annotated-term-left-at-left-border
 =goal>
    ; isa reasoning-task
    phase variation
    step set-moved-focus
    focus nil
    direction left
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =pos
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =pos =loco
    leftborder =pos
 ==>
   !eval! (set-focus =pos *modelfocus*)
 =goal>
    step check-for-initial-annotation
    direction stop
    searchterm nil
    phase inspection
 =imaginal>
)

;; Test persons retrieved the wrong model, so they can't vary the model
;; correctly according to the annotated premise. As a result they press F.
;; Example: Unified Model KMBAP and Annotated-Premise MK with the annotated term K. In this case the unified model can in no case be built with this premise.
(p v-no-focus-set-to-the-position-of-the-annotated-term-left-press-F
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    focus nil
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =pos
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =pos =loco
    leftborder =pos
 ?manual>
    state free
 ==>
=goal>
    phase respond
    step respond-with-f
 =imaginal>
)

;; Discontinuous: If the premise order is discontinuous, because of the merging process the annotated term may not be at the remembered
;; position. Therefore the position of the annoted term has to be found again.
(p v-search-for-the-position-of-the-annotated-term
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    focus nil
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =pos
    loco =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - =pos =loco
    leftborder =lb
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    direction right
    step search-annotated-term
    focus =lb
    searchterm nil
 =retrieval>
 =imaginal>
)


;; Discontinuous: As long as the focus is not at the right border of the current mental model there may be further terms to be checked
;; for annotations to the right. We know that there must be an annotated term. However, we might fail to find it because  due to an
;; erroneous retrieval a model without the correct annotaion or without any annotations might be in the retrieval buffer.
(p v-search-for-the-position-of-the-annotated-term-focus-not-at-right-border-of-current-model-no-match
 =goal>
    ; isa reasoning-task
    phase variation
    step search-annotated-term
    focus =focus
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    loco =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - =focus =loco
    - rightborder =focus
 ==>
   !bind! =newfocus (first (move-focus-right *modelfocus*))
 =goal>
    focus =newfocus
 =retrieval>
 =imaginal>
)



;; Discontinuous: The focus has reached the right border but no annotated term has been found, therfore press the button for the not correct term
(p v-search-for-the-position-of-the-annotated-term-focus-at-right-border-of-current-model-no-match
 =goal>
    ; isa reasoning-task
    phase variation
    step search-annotated-term
    focus =focus
    trial =trial
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    loco =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - =focus =loco
    rightborder =focus
 ==>
=goal>
    phase respond
    step respond-with-f
 =imaginal>
)

;; Discontinuous: An annotated term has been found and focus is on it.  Value in slot focus is set to nil
;; in order to trigger production  v-set-focus-to-the-position-of-the-annotated-term-{right|left} which calls
;; an {inc|dec}-pos.
(p v-search-for-the-position-of-the-annotated-term-match
 =goal>
    ; isa reasoning-task
    phase variation
    step search-annotated-term
    focus =focus
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    loco =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =focus =loco
 ==>
=goal>
    focus nil
    step variation
 =retrieval>
    position =focus
 =imaginal>
)

;; STEP 3
;; The substitution can not be executed, because it contradicts the annotated premise.
;; Try to retrieve another initial annotated premise.
(p v-switch-right-term-with-annotated-term-switch-not-allowed
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    variation yes
    searchterm nil
    - focus nil
    focus =rightposition
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =leftposition
    refo =rightterm
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-for-initial-annotation
    direction stop
    searchterm nil
    phase inspection
 =imaginal>
)

;; STEP 3
;; The substitution can not be executed, because the reference term is the outer left term.
;; Try to retrieve another initial annotated premise.
(p v-switch-right-term-with-annotated-term-switch-not-allowed-at-left-border
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    variation yes
    searchterm nil
    focus =rightposition
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =rightposition
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    refo =rightterm
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =rightposition *modelfocus*)
 =goal>
    step check-for-initial-annotation
    direction stop
    searchterm nil
    phase inspection
 =imaginal>
)

;; The annotated term is substituted with the neighbouring term to the right.
(p v-switch-right-term-with-annotated-term-not-at-rightborder-pmm
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    searchterm nil
    - focus nil
    focus =rightposition
    strategy pmm
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    - rightborder =rightposition
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =leftposition
    - refo =rightterm
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =leftterm
    model am
 +imaginal> =imaginal
)

;; The annotated term is substituted with the neighbouring term to the right.
(p v-switch-right-term-with-annotated-term-not-at-rightborder-any
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    searchterm nil
    - focus nil
    focus =rightposition
    strategy any
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    - rightborder =rightposition
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =leftposition
    - refo =rightterm
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =leftterm
    model am1
 +imaginal> =imaginal
)

;; The annotated term is substituted with the neighbouring term to the right.
(p v-switch-right-term-with-annotated-term-at-rightborder-pmm
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    searchterm nil
    - focus nil
    focus =rightposition
    strategy pmm
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rightposition
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =leftposition
    - refo =rightterm
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =leftterm
    rightborderterm =leftterm
    model am
 +imaginal> =imaginal
)

(p v-switch-right-term-with-annotated-term-at-rightborder-any
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    searchterm nil
    - focus nil
    focus =rightposition
    strategy any
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rightposition
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =leftposition
    - refo =rightterm
    loco =loco
    leftborder =leftannotatedpremiseborder
    =leftannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =leftterm
    rightborderterm =leftterm
    model am1
 +imaginal> =imaginal
)

;; The substitution can not be executed, because it contradicts the annotated premise.
;; Try to retrieve another initial annotated premise.
(p v-switch-left-term-with-annotated-term-switch-not-allowed
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    variation yes
    searchterm nil
    - focus nil
    focus =leftposition
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-for-initial-annotation
    direction stop
    searchterm nil
    phase inspection
 =imaginal>
)

;; STEP 3
;; The substitution can not be executed, because the reference term is the outer right term.
;; Try to retrieve another initial annotated premise.
(p v-switch-left-term-with-annotated-term-switch-not-allowed-at-right-border
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    variation yes
    searchterm nil
    focus =leftposition
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    rightborder =leftposition
    =leftposition =leftterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =leftposition *modelfocus*)
 =goal>
    step check-for-initial-annotation
    direction stop
    searchterm nil
    phase inspection
 =imaginal>
)

;; The annotated term is substituted with the neighbouring term to the left.
(p v-switch-left-term-with-annotated-term-not-at-leftborder-pmm
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    searchterm nil
    - focus nil
    focus =leftposition
    strategy pmm
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - leftborder =leftposition
    leftborder =lb
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    - refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =leftterm
    model am
 +imaginal> =imaginal
)

(p v-switch-left-term-with-annotated-term-not-at-leftborder-any
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    searchterm nil
    - focus nil
    focus =leftposition
    strategy any
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - leftborder =leftposition
    leftborder =lb
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    - refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =leftterm
    model am1
 +imaginal> =imaginal
)

;; The annotated term is substituted with the neighbouring term to the left.
(p v-switch-left-term-with-annotated-term-at-leftborder-pmm
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    searchterm nil
    - focus nil
    focus =leftposition
    strategy pmm
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =leftposition
    leftborder =lb
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    - refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =leftterm
    leftborderterm =rightterm
    model am
 +imaginal> =imaginal
)

(p v-switch-left-term-with-annotated-term-at-leftborder-any
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    searchterm nil
    - focus nil
    focus =leftposition
    strategy any
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =leftposition
    leftborder =lb
    =leftposition =leftterm
    =rightposition =rightterm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    - refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =leftterm
    leftborderterm =rightterm
    model am1
 +imaginal> =imaginal
)

;; STEP 4
;; The substitution was successful. Continue with the comparing/inspection phase.
(p v-switch-done-continue-with-inspection-phase
 =goal>
    ; isa reasoning-task
    phase variation
    step check-variation
    variation yes
    trial =trial
    annotations =annotations
    searchterm =term
    direction =direction
    conclusionsize =cs
    leftborderterm =lbtc
    rightborderterm =rbtc
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lbp
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
 ==>
   !eval! (set-focus =lbp *modelfocus*)
 =goal>
    phase variation
    step recall-unified-model
    trial =trial
    annotations =annotations
    searchterm =term
    direction =direction
    focus nil
    second-focus nil
 +retrieval>
    ; isa mentalmodel
    modeltype conclusion
    modelsize =cs
    leftborderterm =lbtc
    rightborderterm =rbtc
 =imaginal>
    trial =trial
)

;; Retrieval of the conclusion was not successful. Try again.
;; former v-switch-done-continue-with-inspection-phase-retry
(p v-switch-done-continue-with-inspection-phase-retrieval-failure-respond-with-f
 =goal>
    ; isa reasoning-task
    phase variation
    step recall-unified-model
    variation yes
    trial =trial
    annotations =annotations
    direction =direction
    conclusionsize =cs
    leftborderterm =lbtc
    rightborderterm =rbtc
    focus nil
    second-focus nil
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ?retrieval>
    - state busy
    state error
 ==>
=goal>
    phase respond
    step respond-with-f
 =imaginal>
)

;; Conclusion was successfully retrieved. Now retrieve unified model for continuing with the comparision phase.
(p v-retrieved-conclusion-recall-unified-model-pmm
 =goal>
    ; isa reasoning-task
    phase variation
    step recall-unified-model
    trial =trial
    strategy pmm
 =retrieval>
    ; isa mentalmodel
    modeltype conclusion
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 ==>
=goal>
    phase inspection
    step compare-model-with-conclusion
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 +imaginal> =retrieval
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
    trial =trial
    model am
)

(p v-retrieved-conclusion-recall-unified-model-any
 =goal>
    ; isa reasoning-task
    phase variation
    step recall-unified-model
    trial =trial
    strategy any
 =retrieval>
    ; isa mentalmodel
    modeltype conclusion
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 ==>
=goal>
    phase inspection
    step compare-model-with-conclusion
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
 +imaginal> =retrieval
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    leftborderterm =lbt
    rightborderterm =rbt
    trial =trial
    - model pmm
    - model nil
)

;; Retrieval of unified model was not successful. Try again.
(p v-retrieved-conclusion-recall-unified-model-retry-pmm
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    leftborder =lb
    rightborder =rb
    trial =trial
    variation yes
    strategy pmm
 ?retrieval>
    state error
 ?imaginal>
    - buffer empty
    - state busy
 ==>
=goal>
    phase inspection
    step compare-model-with-conclusion
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    trial =trial
    model am
)

(p v-retrieved-conclusion-recall-unified-model-retry-any
 =goal>
    ; isa reasoning-task
    phase inspection
    step compare-model-with-conclusion
    leftborder =lb
    rightborder =rb
    trial =trial
    variation yes
    strategy any
 ?retrieval>
    state error
 ?imaginal>
    - buffer empty
    - state busy
 ==>
=goal>
    phase inspection
    step compare-model-with-conclusion
 +retrieval>
    ; isa mentalmodel
    modeltype unified
    leftborder =lb
    rightborder =rb
    trial =trial
    - model pmm
    - model nil
)

;; STEP 4 a and b are done in the conclusion-phase
;; STEP 5
;; The varied model also does not fit to the conclusion, hence an inherited annotated premise will be retrieved for further variations.
(p v-find-inherited-annotation
 =goal>
    ; isa reasoning-task
    phase inspection
    step check-for-initial-annotation
    trial =trial
    searchterm =term
    direction stop
 ?retrieval>
    buffer empty
    - state busy
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ==>
=goal>
    phase variation
    step variation
    focus nil
 =imaginal>
 +retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    type inherited
    refo =term
    ;;trial =trial ; Restrict to current trial
)

;; No further inherited annotated premises can be found, because they (a) can not be remembered,
;; or (b) there are no further annotated premises.
(p v-found-no-inherited-annotation-press-f
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    trial =trial
    - searchterm nil
 ?retrieval>
    state error
 =imaginal>
    ; isa mentalmodel
    modeltype unified
 ==>
=goal>
    phase respond
    step respond-with-f
 =imaginal>
)

;; STEP 7
;; The annotated term is substituted with the neighbouring term to the left.
(p v-switch-left-term-with-annotated-term-variation2-not-at-leftborder-pmm
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    - searchterm nil
    - focus nil
    focus =leftposition
    strategy pmm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - leftborder =leftposition
    leftborder =lb
    - =leftposition =leftterm
    =leftposition =lefttermm
    =rightposition =rightterm
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =lefttermm
    model am
 +imaginal> =imaginal
)

(p v-switch-left-term-with-annotated-term-variation2-not-at-leftborder-any
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    - searchterm nil
    - focus nil
    focus =leftposition
    strategy any
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    - leftborder =leftposition
    leftborder =lb
    - =leftposition =leftterm
    =leftposition =lefttermm
    =rightposition =rightterm
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =lefttermm
    model am2
 +imaginal> =imaginal
)

;; The annotated term is substituted with the neighbouring term to the left.
(p v-switch-left-term-with-annotated-term-variation2-at-leftborder-pmm
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    - searchterm nil
    - focus nil
    focus =leftposition
    strategy pmm
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =leftposition
    leftborder =lb
    - =leftposition =leftterm
    =leftposition =lefttermm
    =rightposition =rightterm
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =lefttermm
    leftborderterm =rightterm
    model am
 +imaginal> =imaginal
)

(p v-switch-left-term-with-annotated-term-variation2-at-leftborder-any
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    - searchterm nil
    - focus nil
    focus =leftposition
    strategy any
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    leftborder =leftposition
    leftborder =lb
    - =leftposition =leftterm
    =leftposition =lefttermm
    =rightposition =rightterm
 ==>
   !eval! (set-focus =lb *modelfocus*)
 =goal>
    step check-variation
    direction stop
    searchterm =loco
    focus =lb
 =retrieval>
 =imaginal>
    =leftposition =rightterm
    =rightposition =lefttermm
    leftborderterm =rightterm
    model am2
 +imaginal> =imaginal
)

;; The annotated term is substituted with the neighbouring term to the right.
(p v-switch-left-term-with-annotated-term-variation2-not-allowed
 =goal>
    ; isa reasoning-task
    phase variation
    step variation
    - searchterm nil
    trial =trial
    - focus nil
    focus =leftposition
    variation yes
 =retrieval>
    ; isa mentalmodel
    modeltype annotated-premise
    position =rightposition
    refo =leftterm
    loco =loco
    rightborder =rightannotatedpremiseborder
    =rightannotatedpremiseborder =loco
 =imaginal>
    ; isa mentalmodel
    modeltype unified
    =leftposition =leftterm
    =rightposition =rightterm
 ?manual>
    state free
 ==>
=goal>
    step check-for-initial-annotation
    phase inspection
    direction stop
    searchterm =loco
 =imaginal>
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; R: RESPOND ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Productions for responding. Depending on the fact if the model has to create two responses, e.g. after the conclusion was shown
;;; a possible model is presented that has also to be compared to the created (probably variated) unified model.


;; Give answer R. No more answers have to be given, so continue with next task.
(p r-respond-with-r-continue-with-next-trial
 =goal>
    ; isa reasoning-task
    phase respond
    step respond-with-r
    trial =trial
 =imaginal>
    ; isa mentalmodel
 ?manual>
    state free
 ==>
+goal>
    ; isa experiment
    state attend
    trial =trial
 +manual>
    isa press-key
    key "R"
 +visual>
    isa clear
)

;; Give answer F. No more answers have to be given, so continue with next task.
(p r-respond-with-f-continue-with-next-trial
 =goal>
    ; isa reasoning-task
    phase respond
    step respond-with-f
    trial =trial
 =imaginal>
    ; isa mentalmodel
 ?manual>
    state free
 ==>
+goal>
    ; isa experiment
    state attend
    trial =trial
 +manual>
    isa press-key
    key "F"
 +visual>
    isa clear
)





 (setf *actr-enabled-p t)
 )
