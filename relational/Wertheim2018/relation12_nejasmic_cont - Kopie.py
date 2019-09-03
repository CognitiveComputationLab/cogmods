import nengo
import nengo_spa as spa
import numpy as np
import pytry

class RelationalNejacont(pytry.NengoTrial):
    def params(self):
        self.param('number of dimensions', D=128)
        self.param('number of dimensions for represented space', D_space=128)
        self.param('time to run simulation', T=4.0)

    def model(self, p):
        vocab_space = spa.Vocabulary(p.D_space, strict=False)
        vocab_space.populate('OBJ1; OBJ2; OBJ3; OBJ4; X; Y')

        vocab_rule = spa.Vocabulary(p.D)
        vocab_rule.populate('OBJ1; OBJ2; OBJ3; OBJ4; BELOW; ABOVE; LEFT; RIGHT; S; O; V')

        vocab_obj = vocab_rule.create_subset(['OBJ1', 'OBJ2', 'OBJ3', 'OBJ4'])
        vocab_rel = vocab_rule.create_subset(['BELOW', 'ABOVE', 'LEFT', 'RIGHT'])

        model = spa.Network()
        with model:
            
            rule = spa.State(vocab_rule)
            
            objs = spa.State(feedback=1, vocab=vocab_space)
            
            subject = spa.WTAAssocMem(threshold=0.3, input_vocab=vocab_obj)
            object = spa.WTAAssocMem(threshold=0.3, input_vocab=vocab_obj)
            relation = spa.WTAAssocMem(threshold=0.3, input_vocab=vocab_rel)
            
            status = spa.State(p.D)

            def input(t):
                t = t % 1.2
                if t<0.4:
                    return 'OBJ1*S+LEFT*V+OBJ2*O'
                elif t<0.8:
                    return 'OBJ2*S+LEFT*V+OBJ3*O'
                elif t<1.2:
                    return 'OBJ3*S+LEFT*V+OBJ4*O'
          
            
            input = spa.Transcode(input, output_vocab=vocab_rule)



            speed = 0.7
            separation = 0.7
            strength = 0.5

            spa.Actions('''
                input -> rule
                reinterpret(rule*~S) ->subject
                reinterpret(rule*~O) ->object
                reinterpret(rule*~V) ->relation
                
                ifmax (dot(relation, BELOW-ABOVE-LEFT-RIGHT)*strength +
                       (dot(objs, translate(subject)*Y) - dot(objs, translate(object)*Y)) +
                       separation):
                            BAD -> status
                            speed*Y*translate(object)-speed*Y*translate(subject) -> objs
                elifmax (dot(relation, BELOW-ABOVE-LEFT-RIGHT)*strength -
                       (dot(objs, translate(subject)*Y) - dot(objs, translate(object)*Y)) -
                       separation):
                            GOOD -> status
                            
                ifmax (dot(relation, ABOVE-BELOW-LEFT-RIGHT)*strength +
                       (dot(objs, translate(subject)*Y) - dot(objs, translate(object)*Y)) +
                       separation):
                            BAD -> status
                            speed*Y*translate(object)-speed*Y*translate(subject) -> objs
                elifmax (dot(relation, ABOVE-BELOW-LEFT-RIGHT)*strength -
                       (dot(objs, translate(subject)*Y) - dot(objs, translate(object)*Y)) -
                       separation):
                            GOOD -> status    
                            
                ifmax (dot(relation, LEFT-RIGHT-ABOVE-BELOW)*strength +
                       (dot(objs, translate(subject)*X) - dot(objs, translate(object)*X)) +
                       separation):
                            BAD -> status
                            speed*X*translate(object)-speed*X*translate(subject) -> objs
                elifmax (dot(relation, LEFT-RIGHT-ABOVE-BELOW)*strength -
                       (dot(objs, translate(subject)*X) - dot(objs, translate(object)*X)) -
                       separation):
                            GOOD -> status
                            
                ifmax (dot(relation, RIGHT-ABOVE-BELOW-LEFT)*strength +
                       (dot(objs, translate(subject)*X) - dot(objs, translate(object)*X)) +
                       separation):
                            BAD -> status
                            speed*X*translate(object)-speed*X*translate(subject) -> objs
                elifmax (dot(relation, RIGHT-ABOVE-BELOW-LEFT)*strength -
                       (dot(objs, translate(subject)*X) - dot(objs, translate(object)*X)) -
                       separation):
                            GOOD -> status                        
                ''')
            

            def display_node(t, x):
                return x
            
            display = nengo.Node(display_node, size_in=10)
            
            nengo.Connection(objs.output, display[0], 
                      transform=[vocab_space.parse('OBJ1*X').v])
            nengo.Connection(objs.output, display[1], 
                      transform=[vocab_space.parse('OBJ1*Y').v])
            nengo.Connection(objs.output, display[2], 
                      transform=[vocab_space.parse('OBJ2*X').v])
            nengo.Connection(objs.output, display[3], 
                      transform=[vocab_space.parse('OBJ2*Y').v])
            nengo.Connection(objs.output, display[4], 
                      transform=[vocab_space.parse('OBJ3*X').v])
            nengo.Connection(objs.output, display[5], 
                      transform=[vocab_space.parse('OBJ3*Y').v])              
            nengo.Connection(objs.output, display[6], 
                      transform=[vocab_space.parse('OBJ4*X').v])
            nengo.Connection(objs.output, display[7], 
                      transform=[vocab_space.parse('OBJ4*Y').v])           
                      
        for net in model.networks:
            if net.label is not None:
                if net.label.startswith('dot'):
                    net.label = ''
                if net.label.startswith('channel'):
                    net.label = ''
                    
                    
        with model:
            self.result = nengo.Probe(display, synapse=0.03)
        return model


    def evaluate(self, p, sim, plt):
        sim.run(p.T)

        data = sim.data[self.result]
        final_data = data[-1]
        obj1x, obj1y, obj2x, obj2y, obj3x, obj3y, obj4x, obj4y = final_data
        if obj1x<obj2x<obj3x<obj4x:
            score = 1
        else:
            score = 0

        if plt:
            plt.subplot(2,1,1)
            plt.plot(sim.trange(), data[:,::2])
            plt.legend(loc='best', labels=['1','2','3','4'])
            plt.ylabel('x value')
            plt.subplot(2,1,2)
            plt.plot(sim.trange(), data[:,1::2])
            plt.legend(loc='best', labels=['1','2','3','4'])
            plt.ylabel('y value')

        return dict(score=score)
