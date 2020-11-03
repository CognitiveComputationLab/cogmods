from modular_models.models.basic_models.mreasoner import MReasoner
from modular_models.models.ccobra_models.ccobra_mreasoner import CCobraMReasoner
from ccobra import Item
from modular_models.util.sylutil import GENERIC_ITEMS
from ccobra.syllogistic_generalized import GeneralizedSyllogisticEncoder

if __name__ == '__main__':
    # Basic call to the MReasoner without the CCobra wrapper
    m = MReasoner()
    print(m.predict('AA1'))

    # Encode something
    encoder = GeneralizedSyllogisticEncoder()
    raw_task = [['Some', 'a', 'b'], ['All', 'b', 'c']]
    # Create a "Few" task.
    # Note: Right now it returns the wrong result because I made the implementation 'just work',
    # I didn't invest much time in the logic itself. Note that the symbols "T" and "B".
    raw_task = [['Few', 'a', 'b'], ['All', 'b', 'c']]
    # Encode task that supports "most" and "few"
    task = encoder.encode_task(raw_task)
    print('Task: {}'.format(task))
    print('Prediction: {}'.format(m.predict(task)[0]))
    exit(0)

    # Call with CCobra Wrapper
    cobra_m = CCobraMReasoner()
    # Pick a generic ccobra.Item
    item = GENERIC_ITEMS[0]
    print(cobra_m.predict(item))

    # Creating a sample item by hand
    item2 = Item(
        task='All;a;b/All;b;c',
        choices='nothing|not;a',
        resp_type='single-choice',
        domain='syllogistic',
        identifier=0,
        sequence_number=0
    )
    print(cobra_m.predict(item2))

    # Creating another sample item by hand
    # TODO: Implement "Few -> F"
    # TODO: Implement "Most -> M"
    item3 = Item(
        task='All;a;b/Few;b;c',
        choices='nothing|not;a|few;a;b',
        resp_type='single-choice',
        domain='syllogistic',
        identifier=0,
        sequence_number=0
    )
    print(cobra_m.predict(item3))
